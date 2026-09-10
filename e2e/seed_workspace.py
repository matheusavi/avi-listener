"""Build a throwaway workspace for the end-to-end tests.

The dashboard is a view over a folder of projects and meetings, so the way to
test it without a GPU, a microphone or a Whisper model is to write that folder
directly. Every meeting here is a state the interface has to handle: nothing
recorded, recorded but not transcribed, and ready to merge.

Takes the workspace directory as its only argument and writes nothing outside
it, so a test run can never reach the workspace the user keeps real meetings
in.
"""

from __future__ import annotations

import json
import math
import shutil
import struct
import sys
import wave
from datetime import datetime, timezone
from pathlib import Path

# The session start is carried in the continuous recording's filename: that is
# what puts diarized seconds and microphone timestamps on one clock.
SESSION_STEM = "2026-08-24T15-48-48-732Z-system-continuous"
SESSION_START = datetime.strptime("2026-08-24T15-48-48-732", "%Y-%m-%dT%H-%M-%S-%f").replace(
    tzinfo=timezone.utc
).timestamp()

SAMPLE_RATE = 16000
SESSION_SECONDS = 14


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_wav(path: Path, seconds: float, frequency: float = 220.0) -> None:
    """A quiet tone, so speaker samples can really be cut out of it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = bytearray()
    for index in range(int(seconds * SAMPLE_RATE)):
        value = int(6000 * math.sin(2 * math.pi * frequency * index / SAMPLE_RATE))
        frames += struct.pack("<h", value)
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(SAMPLE_RATE)
        writer.writeframes(bytes(frames))


# Relative to the start of the continuous recording, as diarization reports it.
TURNS = [
    {"start": 0.5, "end": 4.0, "speaker": "speaker_0", "text": "Bom dia a todos, vamos comecar."},
    {"start": 4.5, "end": 7.5, "speaker": "speaker_1", "text": "Consigo ouvir sem problemas."},
    {"start": 8.0, "end": 11.5, "speaker": "speaker_0", "text": "Otimo, entao seguimos com a pauta."},
]

# The microphone is transcribed per utterance on the absolute clock, and is the
# other half of what the merge step has to interleave.
MIC_EVENTS = [
    {"offset": 7.6, "duration": 1.4, "text": "Tudo certo por aqui tambem."},
    {"offset": 12.0, "duration": 1.6, "text": "Combinado, obrigado pessoal."},
]


def seed_recording(meeting: Path) -> None:
    write_wav(meeting / "recordings" / "continuous" / f"{SESSION_STEM}.wav", SESSION_SECONDS)
    # One clip per utterance is what the recorder leaves beside the session
    # file; the interface counts them per source.
    for index in (1, 2):
        write_wav(meeting / "recordings" / f"2026-08-24T15-48-5{index}-000Z-mic-{index}.wav", 0.4)


def seed_transcripts(meeting: Path) -> None:
    directory = meeting / "transcripts"
    directory.mkdir(parents=True, exist_ok=True)
    lines = []
    combined = []
    for event in MIC_EVENTS:
        started = SESSION_START + event["offset"]
        record = {
            "source": "mic",
            "text": event["text"],
            "started_at": started,
            "ended_at": started + event["duration"],
            "rms": 0.02,
        }
        lines.append(json.dumps(record, ensure_ascii=False))
        stamp = datetime.fromtimestamp(started, timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        combined.append(f"[{stamp} - {stamp}] {'mic':<24} | {event['text']}")
    (directory / "events.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (directory / "combined.txt").write_text("\n".join(combined) + "\n", encoding="utf-8")


def seed_diarization(meeting: Path) -> None:
    directory = meeting / "diarized"
    directory.mkdir(parents=True, exist_ok=True)

    # Named after the audio it describes: that is how the merge step recovers
    # which recording, and which session start, the turns belong to.
    rttm = "\n".join(
        f"SPEAKER {SESSION_STEM} 1 {turn['start']:.3f} {turn['end'] - turn['start']:.3f} "
        f"<NA> <NA> {turn['speaker']} <NA> <NA>"
        for turn in TURNS
    )
    (directory / f"{SESSION_STEM}.rttm").write_text(rttm + "\n", encoding="utf-8")

    write_json(
        directory / "diarization.json",
        [{"start": turn["start"], "end": turn["end"], "speaker": turn["speaker"]} for turn in TURNS],
    )
    write_json(
        directory / "events.json",
        [
            {
                "start": turn["start"],
                "end": turn["end"],
                "speaker": turn["speaker"],
                "speaker_label": turn["speaker"],
                "text": turn["text"],
            }
            for turn in TURNS
        ],
    )
    (directory / "combined.md").write_text(
        "\n".join(f"**{turn['speaker']}**: {turn['text']}" for turn in TURNS) + "\n",
        encoding="utf-8",
    )


def seed_logs(meeting: Path) -> None:
    directory = meeting / "logs"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "2026-08-24T15-48-48-recording.log").write_text(
        "[2026-08-24 15:48:48] recording started: mic, system\n"
        "[2026-08-24 15:49:02] recording stopped\n",
        encoding="utf-8",
    )


def seed_second_log(meeting: Path) -> None:
    """A second log, so the picker that chooses between them has something to
    choose. One log hides the control entirely."""
    directory = meeting / "logs"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "2026-08-24T16-02-11-transcribe.log").write_text(
        "[2026-08-24 16:02:11] transcribe started\n"
        "[2026-08-24 16:02:40] Transcribed 2 file(s)\n",
        encoding="utf-8",
    )


def create_meeting(project: Path, slug: str, name: str, created_at: float, config: dict | None = None) -> Path:
    meeting = project / slug
    meeting.mkdir(parents=True, exist_ok=True)
    write_json(
        meeting / "meeting.json",
        {"name": name, "created_at": created_at, "config": config or {}},
    )
    return meeting


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: seed_workspace.py <workspace-dir>")

    root = Path(sys.argv[1]).resolve()
    # A fresh workspace every run: a test that depends on what a previous run
    # left behind passes for the wrong reason.
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    base_config = {
        "language": "pt",
        "me": "host",
        # CPU and a small model: nothing here loads a model, but a fixture
        # that implies a GPU invites someone to run it.
        "device": "cpu",
        "model_size": "base",
        "diarization_engine": "nemo",
    }

    project = root / "e2e-project"
    project.mkdir(parents=True, exist_ok=True)
    write_json(project / "project.json", {"name": "E2E Project", "config": dict(base_config)})

    # Settings are per project, and the tests that change them would otherwise
    # be editing the same config the meeting tests read. Each gets its own.
    for slug, name in (("settings-project", "Settings Project"), ("token-project", "Token Project")):
        other = root / slug
        other.mkdir(parents=True, exist_ok=True)
        write_json(other / "project.json", {"name": name, "config": dict(base_config)})

    create_meeting(project, "empty-meeting", "Empty Meeting", 100.0)

    recorded = create_meeting(project, "recorded-meeting", "Recorded Meeting", 200.0)
    seed_recording(recorded)
    seed_logs(recorded)

    logs_meeting = create_meeting(project, "logs-meeting", "Logs Meeting", 150.0)
    seed_recording(logs_meeting)
    seed_logs(logs_meeting)
    seed_second_log(logs_meeting)

    # Two meetings ready to merge, so the tests that change one cannot affect
    # the other.
    for slug, name, created in (
        ("merge-meeting", "Merge Meeting", 300.0),
        ("speakers-meeting", "Speakers Meeting", 400.0),
    ):
        meeting = create_meeting(project, slug, name, created)
        seed_recording(meeting)
        seed_transcripts(meeting)
        seed_diarization(meeting)
        seed_logs(meeting)

    # Actual compact audio and mapping, without loading any speech models.
    from avilistener.meeting import DiarizationTurn, _emit_speaker_lines, write_turns
    from avilistener.recorder import continuous_filename
    from avilistener.timeline import build_timeline, inspect_parts
    resumed = create_meeting(project, "resumed-meeting", "Resumed Meeting", 500.0)
    paths = [resumed / "recordings" / "continuous" / continuous_filename(SESSION_START + offset, "chrome")
             for offset in (0, 1000)]
    for audio in paths:
        write_wav(audio, 4)
    directory = resumed / "diarized"
    build_timeline(paths, directory, inspect_parts(paths))
    turns = [DiarizationTurn(0, 4, "speaker_0"), DiarizationTurn(4, 8, "speaker_0")]
    write_turns(directory / "diarization.json", turns)
    (directory / "session-audio.rttm").write_text(
        "SPEAKER session-audio 1 0 4 <NA> <NA> speaker_0 <NA> <NA>\n"
        "SPEAKER session-audio 1 4 4 <NA> <NA> speaker_0 <NA> <NA>\n", encoding="utf-8",
    )
    _emit_speaker_lines([(1, 2, "Before interruption"), (5, 6, "After resuming")], [], turns, directory, {})
    from avilistener.transcriber import TranscriptResult
    from avilistener.writer import TranscriptWriter
    writer = TranscriptWriter(resumed / "transcripts")
    for offset in (0, 1000):
        writer.write(TranscriptResult("chrome", "duplicate clip", SESSION_START + offset + 1, SESSION_START + offset + 2, .1))
        writer.write(TranscriptResult("mic", f"Microphone {offset}", SESSION_START + offset + 2, SESSION_START + offset + 3, .1))

    print(f"seeded {root}")


if __name__ == "__main__":
    main()
