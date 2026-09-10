"""The real dashboard on top of a pretend machine, for screenshots and GIFs.

Everything that needs hardware or a model is swapped for a stand-in that takes
a couple of seconds and produces plausible output: the microphone and loopback
recorder, the Whisper model, the speaker-splitting engine, and the Discord
receiver. The API, the workspace on disk, the processing pipeline (staging,
timeline, publish), the merge step and the interface are the real ones, so what
the pictures show is what the dashboard does.

Every name, device and line of dialogue here is invented. The workspace is a
throwaway folder in the system temp directory.

    python e2e/demo/demo_server.py [port]
"""

from __future__ import annotations

import json
import math
import os
import random
import shutil
import struct
import sys
import tempfile
import threading
import time
import wave
from datetime import datetime, timezone
from pathlib import Path

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8124
WORKSPACE = Path(tempfile.gettempdir()) / "avilistener-demo-workspace"
SAMPLE_RATE = 16000

# Must be in place before the app module reads it: the interface only offers
# Discord when a token exists. The stand-in receiver never uses it.
os.environ["AVILISTENER_WORKSPACE"] = str(WORKSPACE)
os.environ["DISCORD_BOT_TOKEN"] = "demo-token-never-used"
os.environ["HF_TOKEN"] = "demo-token-never-used"

if WORKSPACE.exists():
    shutil.rmtree(WORKSPACE)
WORKSPACE.mkdir(parents=True)

import avilistener.audio as audio_module  # noqa: E402
import avilistener.meeting as meeting_module  # noqa: E402
import avilistener.server.app as app_module  # noqa: E402
import avilistener.server.jobs as jobs_module  # noqa: E402
import avilistener.transcriber as transcriber_module  # noqa: E402
from avilistener.file_transcriber import source_name_from_discord_wav  # noqa: E402
from avilistener.server.jobs import append_log  # noqa: E402
from avilistener.transcriber import TranscriptResult  # noqa: E402

# ---- the script ------------------------------------------------------------

# The shared PC audio, as the speaker-splitting engine would report it:
# seconds from the start of the session, and who was talking.
TURNS = [
    (0.8, 4.6, "speaker_0", "Morning everyone. Let's start with the release."),
    (9.4, 12.2, "speaker_1", "I still see one flaky test on Windows."),
    (12.6, 15.0, "speaker_0", "Let's fix it before we tag, then."),
    (19.0, 21.8, "speaker_1", "Then we can ship on Thursday."),
]

# Seconds into the session at which the host's microphone clips start.
MIC_OFFSETS = [5.0, 15.6, 22.4]

# What each source "said", in order. The transcriber stand-in hands these out
# one per clip, so the merged transcript reads like a conversation. The PC
# audio clips get the same words as the speaker turns; the merge step drops
# them in favour of the speaker-split version of the same audio.
LINES = {
    "mic": [
        "Sure. The build passed last night, so the branch is ready.",
        "I can take the flaky test right after this call.",
        "Thursday works for me.",
    ],
    "system": [turn[3] for turn in TURNS],
    "alice": [
        "Everyone here? Buffs are going out now.",
        "Tank, wait for my mark before you pull.",
        "Nice, that phase went much better than last week.",
    ],
    "bob_the_tank": [
        "Pulling the boss at the count of three.",
        "Healers, keep an eye on the tank during the second phase.",
    ],
    "dave": [
        "Ready. Potions are up.",
        "Nice pull, that was clean.",
    ],
    "erin": [
        "Loot is in the guild bank, roll when you are back.",
    ],
}

# The host during the raid, instead of the team-sync lines above.
RAID_MIC_LINES = [
    "Everyone in position? Pulling in three.",
    "Watch the adds on the left side.",
    "Good run. Same time on Thursday.",
]

DISCORD_USERS = [
    ("alice", "100000000000000001"),
    ("bob_the_tank", "100000000000000002"),
    ("dave", "100000000000000003"),
    ("erin", "100000000000000004"),
]

SESSION_SECONDS = 26

# Where the last recording went, so the transcriber stand-in can tell a raid
# from a team sync and pick the host's lines accordingly.
LAST_RECORDING_DIR: Path | None = None


def stamp(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%dT%H-%M-%S-") + f"{int(ts * 1000) % 1000:03d}Z"


def write_tone(path: Path, seconds: float, frequency: float = 220.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = bytearray()
    for index in range(int(seconds * SAMPLE_RATE)):
        frames += struct.pack("<h", int(5000 * math.sin(2 * math.pi * frequency * index / SAMPLE_RATE)))
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(SAMPLE_RATE)
        writer.writeframes(bytes(frames))


# ---- pretend hardware --------------------------------------------------------


def fake_devices() -> dict:
    return {
        "microphones": [
            {"name": "Headset Microphone (USB Audio Device)", "default": True},
            {"name": "Microphone Array (Built-in)", "default": False},
        ],
        "speakers": [
            {"name": "Headset Earphone (USB Audio Device)", "default": True},
            {"name": "Speakers (High Definition Audio)", "default": False},
        ],
    }


class FakeRecordingManager:
    """Looks like a live recording: meters move, files grow, clip counts rise.
    On stop it writes the files the real recorder would leave behind."""

    def __init__(self) -> None:
        self._session: dict | None = None
        self._log_path: Path | None = None

    @property
    def session(self):
        return self._session

    def status(self) -> dict | None:
        session = self._session
        if session is None:
            return None
        elapsed = time.time() - session["started_at"]
        peaks = {}
        for name in session["sources"]:
            # Speech-like: mostly loud with the odd quiet moment.
            phase = math.sin(elapsed * (1.7 if name == "mic" else 1.1) + hash(name) % 7)
            peaks[name] = round(0.012 + 0.03 * abs(phase) + random.uniform(0, 0.006), 4)
        return {
            "meeting": session["meeting"],
            "running": True,
            "sources": session["sources"],
            "started_at": session["started_at"],
            "elapsed": elapsed,
            "segments": {name: int(elapsed // 2.5) for name in session["sources"]},
            "peaks": peaks,
            "continuous_files": {
                name: {
                    "filename": f"{stamp(session['started_at'])}-{name}-continuous.wav",
                    "bytes": 44 + int(elapsed * SAMPLE_RATE * 2),
                    "state": "growing",
                }
                for name in session["sources"]
            },
            "errors": {},
        }

    def start(self, meeting_key, output_dir: Path, sources, sample_rate, devices, log_dir: Path | None = None) -> dict:
        global LAST_RECORDING_DIR
        if self._session is not None:
            raise RuntimeError("A recording is already running")
        if not sources:
            raise ValueError("Pick at least one source to record")
        output_dir.mkdir(parents=True, exist_ok=True)
        LAST_RECORDING_DIR = output_dir
        self._log_path = (log_dir / f"{time.strftime('%Y-%m-%dT%H-%M-%S')}-recording.log") if log_dir else None
        append_log(self._log_path, f"recording started: {', '.join(sources)}")
        for name in sources:
            append_log(self._log_path, f"  {name} -> {devices.get(name) or 'system default'}")
        self._session = {
            "meeting": meeting_key,
            "output_dir": output_dir,
            "sources": list(sources),
            "started_at": time.time(),
        }
        return self.status()

    def push_browser_audio(self, meeting_key, payload, sample_rate) -> int:
        return 0

    def stop(self) -> dict:
        session = self._session
        if session is None:
            raise RuntimeError("Nothing is recording")
        status = self.status()
        status["running"] = False
        for info in status["continuous_files"].values():
            info["state"] = "complete"
        started = session["started_at"]
        out: Path = session["output_dir"]
        for name in session["sources"]:
            # The session file is what speaker splitting reads; the clips are
            # what transcription reads. The real recorder writes both.
            write_tone(out / "continuous" / f"{stamp(started)}-{name}-continuous.wav", SESSION_SECONDS)
            offsets = [turn[0] for turn in TURNS] if name == "system" else MIC_OFFSETS
            for index, offset in enumerate(offsets, start=1):
                write_tone(out / f"{stamp(started + offset)}-{name}-{index}.wav", 2.0)
            append_log(self._log_path, f"{name}: {len(offsets)} clip(s), {SESSION_SECONDS}s continuous, peak 0.0412")
        append_log(self._log_path, "recording stopped")
        self._session = None
        return status


class FakeDiscordReceiver:
    """The bot, minus Discord: logs what a real session logs and leaves one
    WAV per participant when it stops."""

    def __init__(self) -> None:
        self.running = False
        self.log: list[str] = []
        self.meeting: str | None = None
        self._output: Path | None = None
        self._started = 0.0

    def start(self, output_dir: Path, token: str, channel_id: str | None, meeting: str | None = None) -> None:
        global LAST_RECORDING_DIR
        self._output = output_dir
        self._started = time.time()
        self.meeting = meeting
        self.running = True
        self.log = []
        LAST_RECORDING_DIR = output_dir
        script = [
            (0.0, "Logged in as AviListenerBot"),
            (0.6, f"Joining voice channel {channel_id or '(waiting for !listen)'}"),
            (1.4, "Connected. Receiving audio from 4 participants."),
            (3.0, "alice is speaking"),
            (5.5, "bob_the_tank is speaking"),
            (8.0, "dave is speaking"),
            (10.5, "erin is speaking"),
            (13.0, "bob_the_tank is speaking"),
        ]
        started = self._started

        def run() -> None:
            for delay, line in script:
                time.sleep(max(0.0, started + delay - time.time()))
                if not self.running or self._started != started:
                    return
                self.log.append(f"[{time.strftime('%H:%M:%S')}] {line}")

        threading.Thread(target=run, daemon=True).start()

    def stop(self) -> None:
        if not self.running:
            return
        self.running = False
        out = self._output
        if out is not None:
            offset = 1.0
            for user, user_id in DISCORD_USERS:
                for _ in LINES[user]:
                    write_tone(out / f"{stamp(self._started + offset)}-{user}-{user_id}.wav", 2.4)
                    offset += 3.1
        self.log.append(f"[{time.strftime('%H:%M:%S')}] Left the voice channel. Wrote one file per participant.")

    def status(self) -> dict:
        return {
            "running": self.running,
            "meeting": self.meeting,
            "started_at": self._started or None,
            "elapsed": (time.time() - self._started) if self.running else None,
            "log": self.log[-40:],
        }


def fake_channels(token: str) -> list[dict]:
    return [
        {
            "guild": "Tuesday Raiders",
            "channels": [{"id": "200000000000000001", "name": "Raid voice"}, {"id": "200000000000000002", "name": "Lounge"}],
        },
        {"guild": "Board Game Club", "channels": [{"id": "200000000000000003", "name": "Table 1"}]},
    ]


# ---- pretend models ------------------------------------------------------------


class FakeTranscriber:
    """Hands out scripted lines per source, in clip order, at a believable pace."""

    def __init__(self) -> None:
        self._handed_out: dict[str, int] = {}
        users = {user for user, _ in DISCORD_USERS}
        wavs = list(LAST_RECORDING_DIR.glob("*.wav")) if LAST_RECORDING_DIR else []
        self._raid = any(source_name_from_discord_wav(wav) in users for wav in wavs)

    def transcribe(self, chunk) -> TranscriptResult | None:
        time.sleep(0.45)
        lines = RAID_MIC_LINES if (self._raid and chunk.source == "mic") else LINES.get(chunk.source, ["(inaudible)"])
        index = self._handed_out.get(chunk.source, 0)
        self._handed_out[chunk.source] = index + 1
        return TranscriptResult(
            source=chunk.source,
            text=lines[index % len(lines)],
            started_at=chunk.started_at,
            ended_at=chunk.started_at + 2.4,
            rms=0.03,
        )


def fake_build_transcriber(config: dict) -> FakeTranscriber:
    time.sleep(1.8)
    return FakeTranscriber()


def fake_run_meeting_diarize(
    audio_path: Path,
    config: dict,
    output_dir: Path,
    num_speakers,
    max_speakers,
    nemo_python=None,
    speaker_names=None,
    engine="pyannote",
    hf_token=None,
) -> None:
    """Writes what the real engine writes, named after the audio it was given."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(audio_path).stem
    print(f"Loading {engine} pipeline")
    time.sleep(1.5)
    print(f"Found 2 speakers in {SESSION_SECONDS}s of audio")
    time.sleep(1.0)
    print("Transcribing each speaker turn")
    time.sleep(1.2)
    rttm = "\n".join(
        f"SPEAKER {stem} 1 {start:.3f} {end - start:.3f} <NA> <NA> {speaker} <NA> <NA>" for start, end, speaker, _ in TURNS
    )
    (output_dir / f"{stem}.rttm").write_text(rttm + "\n", encoding="utf-8")
    (output_dir / "diarization.json").write_text(
        json.dumps([{"start": s, "end": e, "speaker": sp} for s, e, sp, _ in TURNS], indent=2), encoding="utf-8"
    )
    (output_dir / "events.json").write_text(
        json.dumps([{"start": s, "end": e, "speaker": sp, "speaker_label": sp, "text": t} for s, e, sp, t in TURNS], indent=2),
        encoding="utf-8",
    )
    (output_dir / "combined.md").write_text("\n".join(f"**{sp}**: {t}" for _, _, sp, t in TURNS) + "\n", encoding="utf-8")
    (output_dir / "words.json").write_text("[]", encoding="utf-8")


# ---- wire it in -------------------------------------------------------------

audio_module.list_devices_structured = fake_devices
jobs_module.list_voice_channels = fake_channels
transcriber_module.build_transcriber = fake_build_transcriber
meeting_module.run_meeting_diarize = fake_run_meeting_diarize
app_module.recorder = FakeRecordingManager()
app_module.discord = FakeDiscordReceiver()
app_module.pyannote_available = lambda: {"ok": True, "reason": None, "token_in_env": True}

if __name__ == "__main__":
    import uvicorn

    print(f"demo workspace: {WORKSPACE}")
    uvicorn.run(app_module.app, host="127.0.0.1", port=PORT, log_level="warning")
