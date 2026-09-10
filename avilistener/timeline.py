"""Read-only recording inputs, compact working audio, and wall-clock mapping."""
from __future__ import annotations

import hashlib
import json
import wave
from pathlib import Path

import numpy as np

from avilistener.recorder import source_name_from_continuous_wav, timestamp_from_continuous_wav


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def recording_manifest(directory: Path) -> list[dict]:
    return [
        {"path": path.relative_to(directory).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in sorted(directory.rglob("*.wav"))
    ]


def inspect_parts(paths: list[Path]) -> list[dict]:
    """Fail visibly on unreadable audio rather than quietly omitting a part."""
    parts = []
    cursor = 0
    for path in sorted(paths):
        started = timestamp_from_continuous_wav(path)
        source = source_name_from_continuous_wav(path)
        if started is None or source is None:
            raise ValueError(f"Recording has no source/start timestamp: {path.name}")
        before = path.stat()
        digest = sha256(path)
        with wave.open(str(path), "rb") as reader:
            rate, frames = reader.getframerate(), reader.getnframes()
            if reader.getsampwidth() != 2 or reader.getcomptype() != "NONE" or frames <= 0:
                raise ValueError(f"Invalid or unfinished PCM16 WAV: {path.name}")
            # Check actual payload length, not just the header's claimed duration.
            reader.setpos(max(0, frames - 1))
            if len(reader.readframes(1)) != 2 * reader.getnchannels():
                raise ValueError(f"Truncated WAV: {path.name}; preserve the original for recovery")
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError(f"Recording is still changing: {path.name}. Stop recording first.")
        duration = frames / rate
        output_frames = round(duration * 16000)
        parts.append({
            "filename": path.name, "source": source, "sha256": digest,
            "bytes": before.st_size, "sample_rate": rate, "frames": frames,
            "started_at": started, "ended_at": started + duration,
            "offset": cursor / 16000, "duration": output_frames / 16000,
        })
        cursor += output_frames
    if not parts:
        raise ValueError("No shared audio recordings found")
    return parts


def build_timeline(paths: list[Path], output_dir: Path, parts: list[dict]) -> Path:
    """Stream parts into mono 16 kHz audio without adding the unrecorded gaps.

    Originals are only opened in rb mode. The manifest keeps a separate mapping
    per part, including when capture switched between PC audio and Chrome.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    audio = output_dir / "session-audio.wav"
    if audio.resolve() in {p.resolve() for p in paths}:
        raise ValueError("Working audio must not overwrite an original recording")
    by_name = {p.name: p for p in paths}
    with wave.open(str(audio), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        for part in parts:
            path = by_name[part["filename"]]
            with wave.open(str(path), "rb") as reader:
                rate, channels = reader.getframerate(), reader.getnchannels()
                remaining = reader.getnframes()
                while remaining:
                    count = min(rate, remaining)
                    raw = reader.readframes(count)
                    if len(raw) != count * channels * 2:
                        raise ValueError(f"Truncated WAV: {path.name}")
                    if rate == 16000 and channels == 1:
                        writer.writeframesraw(raw)
                    else:
                        samples = np.frombuffer(raw, dtype="<i2").astype(np.float32)
                        samples = samples.reshape(-1, channels).mean(axis=1)
                        # Full-second blocks prevent rounding drift across a long file.
                        target = round(count * 16000 / rate)
                        positions = np.arange(target) * rate / 16000
                        converted = np.interp(positions, np.arange(count), samples)
                        writer.writeframesraw(np.rint(converted).clip(-32768, 32767).astype("<i2").tobytes())
                    remaining -= count
            if sha256(path) != part["sha256"]:
                raise ValueError(f"Recording changed while processing: {path.name}")
    (output_dir / "timeline.json").write_text(json.dumps({
        "version": 1, "audio": audio.name, "sample_rate": 16000, "parts": parts,
    }, indent=2), encoding="utf-8")
    return audio


def load_timeline(directory: Path) -> dict | None:
    path = directory / "timeline.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def part_at(parts: list[dict], seconds: float) -> int:
    for i, part in enumerate(parts):
        if seconds < part["offset"] + part["duration"]:
            return i
    return len(parts) - 1


def wall_time(part: dict, seconds: float) -> float:
    relative = min(max(seconds - part["offset"], 0.0), part["duration"])
    return part["started_at"] + relative
