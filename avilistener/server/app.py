"""HTTP API and static host for the AviListener dashboard.

The endpoints mirror the flow that is actually used: record, transcribe,
diarize the loopback, then merge both sides into one transcript. Each meeting
reports which of those it is currently able to do, so the interface can never
offer a step before the thing it consumes exists.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from avilistener.server.jobs import DiscordReceiver, JobRegistry, RecordingManager, discord_token_in_environment
from avilistener.server.workspace import DEFAULT_PROJECT_CONFIG, Meeting, Workspace, public_config

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
# Overridable so a test run gets its own workspace: without this the only
# place meetings can live is beside the checkout, and an end-to-end run would
# create projects in the one the user actually keeps recordings in.
WORKSPACE_ROOT = Path(os.environ.get("AVILISTENER_WORKSPACE") or PROJECT_ROOT / "workspace")
WEB_DIST = PROJECT_ROOT / "web" / "dist"

workspace = Workspace(root=WORKSPACE_ROOT)
jobs = JobRegistry()
recorder = RecordingManager()
discord = DiscordReceiver(project_root=PROJECT_ROOT)

app = FastAPI(title="AviListener")
app.add_middleware(
    CORSMiddleware,
    # The Vite dev server runs on another port; the API only ever binds to
    # localhost, so this stays local either way.
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ProjectIn(BaseModel):
    name: str
    config: dict[str, Any] | None = None


class MeetingIn(BaseModel):
    name: str
    config: dict[str, Any] | None = None


class RecordIn(BaseModel):
    sources: list[str] = ["mic", "system"]


class DiarizeIn(BaseModel):
    num_speakers: int | None = None
    max_speakers: int = 6
    engine: str | None = None  # falls back to the project's diarization_engine


class CombineIn(BaseModel):
    me: str | None = None


class ConfigIn(BaseModel):
    config: dict[str, Any]


def _meeting_or_404(project_slug: str, meeting_slug: str) -> Meeting:
    project = workspace.project(project_slug)
    if project is None:
        raise HTTPException(404, f"No project {project_slug}")
    meeting = project.meeting(meeting_slug)
    if meeting is None:
        raise HTTPException(404, f"No meeting {meeting_slug}")
    return meeting


def _key(meeting: Meeting) -> str:
    return f"{meeting.project.slug}/{meeting.slug}"


def _discord_status_for(meeting: Meeting) -> dict:
    """The receiver is one process for the whole dashboard, so another
    meeting must not see it as its own recording."""
    status = discord.status()
    if status.get("meeting") not in (None, _key(meeting)):
        return {"running": False, "meeting": None, "started_at": None, "elapsed": None, "log": []}
    return status


def _transcription_config(meeting: Meeting) -> dict:
    from avilistener.processing import transcription_config
    return transcription_config(meeting)


def _require_stopped(meeting: Meeting) -> None:
    status = recorder.status()
    if status and status.get("running"):
        raise HTTPException(409, "Stop recording before processing audio")
    if discord.running:
        raise HTTPException(409, "Stop Discord recording before processing audio")


@app.get("/api/health")
def health() -> dict:
    return {"ok": True}


@app.get("/api/devices")
def devices() -> dict:
    from avilistener.audio import list_devices_structured

    try:
        return list_devices_structured()
    except Exception as exc:
        return {"microphones": [], "speakers": [], "error": str(exc)}


@app.get("/api/projects/{project_slug}/discord/channels")
def discord_channels(project_slug: str) -> dict:
    from avilistener.server.jobs import list_voice_channels

    project = workspace.project(project_slug)
    if project is None:
        raise HTTPException(404, f"No project {project_slug}")
    try:
        return {"servers": list_voice_channels(str(project.config.get("discord_token") or ""))}
    except RuntimeError as exc:
        return {"servers": [], "error": str(exc)}


def pyannote_available() -> dict:
    """Whether the pyannote engine can run. The token can also come from a
    project's stored config, which this endpoint cannot see - the UI combines
    both signals."""
    venv_ok = (PROJECT_ROOT / ".venv-pyannote" / "Scripts" / "python.exe").exists()
    token_in_env = bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
    reason = None if venv_ok else "missing .venv-pyannote"
    return {"ok": venv_ok, "reason": reason, "token_in_env": token_in_env}


@app.get("/api/defaults")
def defaults() -> dict:
    return {
        "config": DEFAULT_PROJECT_CONFIG,
        "discord_token_in_env": discord_token_in_environment(),
        "pyannote": pyannote_available(),
    }


@app.get("/api/projects")
def list_projects() -> dict:
    return {
        "projects": [
            {**project.to_json(), "meetings": [meeting.to_json() for meeting in project.meetings()]}
            for project in workspace.projects()
        ],
        "recording": recorder.status(),
    }


@app.post("/api/projects")
def create_project(payload: ProjectIn) -> dict:
    if not payload.name.strip():
        raise HTTPException(400, "Project needs a name")
    project = workspace.create_project(payload.name, payload.config)
    return project.to_json()


@app.put("/api/projects/{project_slug}/config")
def update_project_config(project_slug: str, payload: ConfigIn) -> dict:
    project = workspace.project(project_slug)
    if project is None:
        raise HTTPException(404, f"No project {project_slug}")
    project.update_config(payload.config)
    return project.to_json()


@app.post("/api/projects/{project_slug}/meetings")
def create_meeting(project_slug: str, payload: MeetingIn) -> dict:
    project = workspace.project(project_slug)
    if project is None:
        raise HTTPException(404, f"No project {project_slug}")
    if not payload.name.strip():
        raise HTTPException(400, "Meeting needs a name")
    return project.create_meeting(payload.name, payload.config).to_json()


@app.get("/api/projects/{project_slug}/meetings/{meeting_slug}")
def get_meeting(project_slug: str, meeting_slug: str) -> dict:
    meeting = _meeting_or_404(project_slug, meeting_slug)
    active = jobs.active_for(_key(meeting))
    status = recorder.status()
    return {
        **meeting.to_json(),
        "discord": _discord_status_for(meeting),
        "discord_available": bool(meeting.config.get("discord_token")) or discord_token_in_environment(),
        "recording": status if status and status.get("meeting") == _key(meeting) else None,
        "active_job": active.to_json() if active else None,
        "jobs": [job.to_json() for job in jobs.recent(_key(meeting), limit=8)],
    }


@app.put("/api/projects/{project_slug}/meetings/{meeting_slug}/config")
def update_meeting_config(project_slug: str, meeting_slug: str, payload: ConfigIn) -> dict:
    meeting = _meeting_or_404(project_slug, meeting_slug)
    meeting.update_config(payload.config)
    return meeting.to_json()


@app.get("/api/recording")
def recording_status() -> dict:
    return {"recording": recorder.status()}


@app.post("/api/projects/{project_slug}/meetings/{meeting_slug}/record/start")
def start_recording(project_slug: str, meeting_slug: str, payload: RecordIn) -> dict:
    meeting = _meeting_or_404(project_slug, meeting_slug)
    if jobs.active_for(_key(meeting)):
        raise HTTPException(409, "Wait for processing to finish before resuming recording")
    config = meeting.config
    wants_discord = "discord" in payload.sources
    audio_sources = [source for source in payload.sources if source != "discord"]

    if wants_discord:
        # Started first: if the token is wrong there is no point opening audio
        # devices, and the user gets the real reason straight away.
        try:
            discord.start(
                output_dir=meeting.recordings_dir,
                token=str(config.get("discord_token") or ""),
                channel_id=config.get("discord_channel_id"),
                meeting=_key(meeting),
            )
        except RuntimeError as exc:
            raise HTTPException(409, str(exc))

    status = None
    if audio_sources:
        try:
            status = recorder.start(
                meeting_key=_key(meeting),
                output_dir=meeting.recordings_dir,
                sources=audio_sources,
                sample_rate=int(config.get("sample_rate", 16000)),
                devices=config.get("devices", {}) or {},
                log_dir=meeting.logs_dir,
            )
        except (RuntimeError, ValueError) as exc:
            discord.stop()
            raise HTTPException(409, str(exc))

    meeting.update_config({"sources": payload.sources})
    return {"recording": status, "discord": discord.status()}


@app.post("/api/projects/{project_slug}/meetings/{meeting_slug}/record/stop")
def stop_recording(project_slug: str, meeting_slug: str) -> dict:
    _meeting_or_404(project_slug, meeting_slug)
    discord.stop()
    if recorder.status() is None:
        return {"recording": None, "discord": discord.status()}
    try:
        return {"recording": recorder.stop(), "discord": discord.status()}
    except RuntimeError as exc:
        raise HTTPException(409, str(exc))


@app.post("/api/projects/{project_slug}/meetings/{meeting_slug}/record/chrome")
async def push_chrome_audio(
    project_slug: str,
    meeting_slug: str,
    request: Request,
    sample_rate: int,
) -> dict:
    """Receive one ordered chunk of float32 mono PCM from Chrome tab capture."""
    meeting = _meeting_or_404(project_slug, meeting_slug)
    payload = await request.body()
    try:
        frames = recorder.push_browser_audio(_key(meeting), payload, sample_rate)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except RuntimeError as exc:
        raise HTTPException(409, str(exc))
    return {"frames": frames}


@app.post("/api/projects/{project_slug}/meetings/{meeting_slug}/transcribe")
def transcribe(project_slug: str, meeting_slug: str) -> dict:
    meeting = _meeting_or_404(project_slug, meeting_slug)
    _require_stopped(meeting)
    if not meeting.capabilities()["transcribe"]:
        raise HTTPException(409, "Nothing recorded yet")

    def work(log) -> None:
        from avilistener.processing import transcribe_meeting
        transcribe_meeting(meeting, log)

    try:
        return jobs.start("transcribe", _key(meeting), work, log_dir=meeting.logs_dir).to_json()
    except RuntimeError as exc:
        raise HTTPException(409, str(exc))


@app.post("/api/projects/{project_slug}/meetings/{meeting_slug}/diarize")
def diarize(project_slug: str, meeting_slug: str, payload: DiarizeIn) -> dict:
    meeting = _meeting_or_404(project_slug, meeting_slug)
    _require_stopped(meeting)
    if not meeting.shared_parts():
        raise HTTPException(409, "No loopback recording to diarize")

    engine = payload.engine or str(meeting.config.get("diarization_engine") or "nemo")
    hf_token = str(meeting.config.get("hf_token") or "").strip() or None
    if engine == "pyannote" and not hf_token and not pyannote_available()["token_in_env"]:
        raise HTTPException(
            409,
            "pyannote needs a Hugging Face token: paste one in Settings or set HF_TOKEN before starting the dashboard.",
        )

    def work(log) -> None:
        from avilistener.processing import diarize_meeting
        diarize_meeting(meeting, payload.num_speakers, payload.max_speakers, engine, log)

    meeting.update_config({"num_speakers": payload.num_speakers, "diarization_engine": engine})
    try:
        return jobs.start("diarize", _key(meeting), work, log_dir=meeting.logs_dir).to_json()
    except RuntimeError as exc:
        raise HTTPException(409, str(exc))


@app.post("/api/projects/{project_slug}/meetings/{meeting_slug}/combine")
def combine(project_slug: str, meeting_slug: str, payload: CombineIn) -> dict:
    meeting = _meeting_or_404(project_slug, meeting_slug)
    _require_stopped(meeting)
    if not meeting.capabilities()["combine"]:
        art = meeting.artifacts()
        if art["diarization_stale"] or art["transcription_stale"]:
            raise HTTPException(409, "New recording parts detected. Transcribe and split again before merging.")
        raise HTTPException(409, "Need both a diarized recording and a transcript")

    me = payload.me or meeting.config.get("me") or "host"

    def work(log) -> None:
        _rebuild_merged(meeting)
        events = (meeting.merged_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
        speakers = sorted({json.loads(line)["source"] for line in events if line.strip()})
        log(f"Merged {len(events)} line(s) from {len(speakers)} speaker(s): {', '.join(speakers)}")

    meeting.update_config({"me": me})
    try:
        return jobs.start("combine", _key(meeting), work, log_dir=meeting.logs_dir).to_json()
    except RuntimeError as exc:
        raise HTTPException(409, str(exc))


@app.get("/api/projects/{project_slug}/meetings/{meeting_slug}/logs")
def logs(project_slug: str, meeting_slug: str, name: str | None = None) -> dict:
    """Logs kept beside the meeting they belong to, so they survive a restart."""
    meeting = _meeting_or_404(project_slug, meeting_slug)
    if not meeting.logs_dir.exists():
        return {"files": [], "name": None, "lines": []}

    files = sorted((path.name for path in meeting.logs_dir.glob("*.log")), reverse=True)
    chosen = name if name in files else (files[0] if files else None)
    lines: list[str] = []
    if chosen:
        # Only the tail: a long meeting's log is not something to ship whole.
        lines = (meeting.logs_dir / chosen).read_text(encoding="utf-8", errors="replace").splitlines()[-300:]
    return {"files": files, "name": chosen, "lines": lines}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "No such job")
    return job.to_json()


@app.get("/api/projects/{project_slug}/meetings/{meeting_slug}/transcript")
def transcript(project_slug: str, meeting_slug: str, kind: str = "merged") -> dict:
    meeting = _meeting_or_404(project_slug, meeting_slug)
    sources = {
        "merged": meeting.merged_dir / "combined.txt",
        "segments": meeting.transcripts_dir / "combined.txt",
        "diarized": meeting.diarized_dir / "combined.md",
    }
    path = sources.get(kind)
    if path is None:
        raise HTTPException(400, f"Unknown transcript kind {kind}")
    if not path.exists():
        return {"kind": kind, "exists": False, "lines": []}

    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {"kind": kind, "exists": True, "lines": lines}


@app.get("/api/projects/{project_slug}/meetings/{meeting_slug}/speakers")
def speakers(project_slug: str, meeting_slug: str) -> dict:
    """Detected speakers, with a short sample of each so they can be named."""
    from avilistener.meeting import parse_rttm
    from avilistener.speakers import extract_wav_slice, pick_sample_turn, speaker_labels

    meeting = _meeting_or_404(project_slug, meeting_slug)
    diarization = meeting.diarized_dir / "diarization.json"
    if not diarization.exists():
        return {"speakers": [], "names": {}}

    turns = _turns_from_json(diarization)
    audio = meeting.speaker_audio()
    names = meeting.config.get("speaker_names") or {}
    samples_dir = meeting.path / "speaker-samples"

    speakers_out = []
    for label in speaker_labels(turns):
        spoken = sum(turn.end - turn.start for turn in turns if turn.speaker == label)
        sample = samples_dir / f"{label}.wav"
        if audio is not None and not sample.exists():
            # Cut lazily: a meeting diarized before this existed still gets
            # samples without re-running anything.
            span = pick_sample_turn(turns, label)
            if span is not None:
                extract_wav_slice(audio, sample, span[0], span[1])
        speakers_out.append(
            {
                "label": label,
                "name": names.get(label, ""),
                "seconds": round(spoken, 1),
                "has_sample": sample.exists(),
            }
        )
    return {"speakers": speakers_out, "names": names}


def _turns_from_json(path: Path) -> list:
    from avilistener.meeting import DiarizationTurn

    payload = json.loads(path.read_text(encoding="utf-8"))
    return [DiarizationTurn(start=float(t["start"]), end=float(t["end"]), speaker=str(t["speaker"])) for t in payload]


@app.get("/api/projects/{project_slug}/meetings/{meeting_slug}/speakers/{label}/sample.wav")
def speaker_sample(project_slug: str, meeting_slug: str, label: str) -> FileResponse:
    meeting = _meeting_or_404(project_slug, meeting_slug)
    # Never let a label from the URL escape the meeting's own folder.
    safe = "".join(char for char in label if char.isalnum() or char in "._-")
    sample = meeting.path / "speaker-samples" / f"{safe}.wav"
    if not sample.exists():
        raise HTTPException(404, "No sample for that speaker")
    return FileResponse(sample, media_type="audio/wav")


class SpeakerNamesIn(BaseModel):
    names: dict[str, str]


@app.put("/api/projects/{project_slug}/meetings/{meeting_slug}/speakers")
def rename_speakers(project_slug: str, meeting_slug: str, payload: SpeakerNamesIn) -> dict:
    """Name the speakers, and refresh the merged transcript if there is one.

    Renaming does not touch the clustering, so nothing needs re-running: the
    merged transcript is simply rebuilt with the new labels, which is instant.
    """
    meeting = _meeting_or_404(project_slug, meeting_slug)
    if jobs.active_for(_key(meeting)):
        raise HTTPException(409, "Wait for processing to finish before naming speakers")
    names = {label: name.strip() for label, name in payload.names.items() if name and name.strip()}
    meeting.update_config({"speaker_names": names})

    rebuilt = False
    if meeting.artifacts()["has_merged"] and meeting.capabilities()["combine"]:
        _rebuild_merged(meeting)
        rebuilt = True
    return {"names": names, "merged_updated": rebuilt}


def _rebuild_merged(meeting: Meeting) -> None:
    from avilistener.processing import combine_meeting
    combine_meeting(meeting)


if WEB_DIST.exists():
    app.mount("/assets", StaticFiles(directory=WEB_DIST / "assets"), name="assets")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(WEB_DIST / "index.html")
