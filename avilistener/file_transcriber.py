from __future__ import annotations

import re
import time
import wave
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from avilistener.audio import AudioChunk
from avilistener.transcriber import Transcriber
from avilistener.writer import TranscriptWriter


def transcribe_wav_directory(
    input_dir: str | Path,
    transcriber: Transcriber,
    writer: TranscriptWriter,
    move_processed: bool = True,
    include_processed: bool = False,
) -> int:
    """Transcribe every clip in a directory, oldest first.

    The writer decides where the transcript goes; this only decides what gets
    read and in what order. Order matters: the clips are one conversation, and
    reading them out of order would shuffle it.
    """
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_path}")

    processed_dir = input_path / "processed"
    if move_processed:
        processed_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    wav_paths = collect_wav_clips(input_path, include_processed=include_processed)
    for wav_path in wav_paths:
        chunk = wav_to_audio_chunk(source_name_from_discord_wav(wav_path), wav_path)
        result = transcriber.transcribe(chunk)
        if result is not None:
            writer.write(result)
            count += 1

    if move_processed:
        for wav_path in wav_paths:
            if not wav_path.exists():
                continue
            destination = processed_dir / wav_path.name
            if destination.exists():
                destination = processed_dir / f"{wav_path.stem}-{int(time.time())}{wav_path.suffix}"
            wav_path.replace(destination)

    return count


def collect_wav_clips(input_path: Path, include_processed: bool = True) -> list[Path]:
    """Read both legacy and current clips once, without touching their paths."""
    from avilistener.timeline import sha256

    directories = [input_path]
    if include_processed:
        directories.append(input_path / "processed")
    by_name = {}
    for directory in directories:
        for path in directory.glob("*.wav"):
            if path.name in by_name and sha256(path) != sha256(by_name[path.name]):
                raise ValueError(f"Conflicting recording copies: {path.name}")
            by_name.setdefault(path.name, path)
    return sorted(by_name.values(), key=sort_key_for_discord_wav)


def wav_to_audio_chunk(source_name: str, path: Path) -> AudioChunk:
    with wave.open(str(path), "rb") as f:
        sample_rate = f.getframerate()
        channels = f.getnchannels()
        frames = f.readframes(f.getnframes())

    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape((-1, channels)).mean(axis=1)
    audio = resample_linear(audio, sample_rate, 16000)
    rms = float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0
    duration = len(audio) / 16000 if audio.size else 0.0
    started_at = timestamp_from_discord_wav(path) or (path.stat().st_mtime - duration)
    ended_at = started_at + duration
    return AudioChunk(
        source=source_name,
        audio=audio,
        sample_rate=16000,
        started_at=started_at,
        ended_at=ended_at,
        rms=rms,
    )


def resample_linear(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate or audio.size == 0:
        return audio.astype(np.float32, copy=False)
    duration = audio.shape[0] / source_rate
    target_frames = max(1, int(duration * target_rate))
    old_x = np.linspace(0.0, duration, num=audio.shape[0], endpoint=False)
    new_x = np.linspace(0.0, duration, num=target_frames, endpoint=False)
    return np.interp(new_x, old_x, audio).astype(np.float32)


def source_name_from_discord_wav(path: Path) -> str:
    # Files are timestamp-name-userid.wav; keep the display name portion.
    match = re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}-\d{3}Z-(.+)-(\d+)$", path.stem)
    if not match:
        return path.stem
    return match.group(1).strip() or match.group(2)


def sort_key_for_discord_wav(path: Path) -> tuple[float, str]:
    return (timestamp_from_discord_wav(path) or path.stat().st_mtime, path.name)


def timestamp_from_discord_wav(path: Path) -> float | None:
    match = re.match(r"^(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}-\d{3})Z-", path.stem)
    if not match:
        return None
    try:
        value = datetime.strptime(match.group(1), "%Y-%m-%dT%H-%M-%S-%f").replace(tzinfo=timezone.utc)
        return value.timestamp()
    except ValueError:
        return None
