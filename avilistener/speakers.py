"""Audio samples for identifying who a diarized speaker actually is.

Diarization separates voices but cannot name them, and reading a speaker's
lines is a poor way to recognise someone. A few seconds of their voice settles
it immediately, so each detected speaker gets a short clip to play.
"""

from __future__ import annotations

import wave
from pathlib import Path

from avilistener.meeting import DiarizationTurn


def speaker_labels(turns: list[DiarizationTurn]) -> list[str]:
    """Detected speakers, ordered by how much they spoke."""
    totals: dict[str, float] = {}
    for turn in turns:
        totals[turn.speaker] = totals.get(turn.speaker, 0.0) + (turn.end - turn.start)
    return [speaker for speaker, _ in sorted(totals.items(), key=lambda item: item[1], reverse=True)]


def pick_sample_turn(
    turns: list[DiarizationTurn],
    speaker: str,
    max_seconds: float = 6.0,
    edge_trim: float = 0.25,
) -> tuple[float, float] | None:
    """Pick the clearest few seconds of one speaker.

    Their longest turn is used, because a long turn is the least likely to be
    a misassigned fragment. Both ends are trimmed slightly: diarization
    boundaries are approximate, and the tail of the previous speaker bleeding
    into the sample is exactly what makes identification harder.
    """
    candidates = [turn for turn in turns if turn.speaker == speaker and turn.end > turn.start]
    if not candidates:
        return None

    best = max(candidates, key=lambda turn: turn.end - turn.start)
    start = best.start + edge_trim
    end = min(best.end - edge_trim, start + max_seconds)
    if end <= start:  # turn shorter than the trim; use it whole
        start = best.start
        end = min(best.end, best.start + max_seconds)
    return (start, end)


def extract_wav_slice(source: Path, destination: Path, start: float, end: float) -> bool:
    """Copy one span of a WAV without loading the whole file.

    A session recording can be hours long, so the frames are seeked to and read
    directly rather than decoding everything to reach a few seconds.
    """
    try:
        with wave.open(str(source), "rb") as reader:
            rate = reader.getframerate()
            total = reader.getnframes()
            first = max(0, int(start * rate))
            last = min(total, int(end * rate))
            if last <= first:
                return False

            reader.setpos(first)
            frames = reader.readframes(last - first)
            channels = reader.getnchannels()
            width = reader.getsampwidth()
    except (wave.Error, EOFError, OSError, ValueError):
        return False

    if not frames:
        return False

    destination.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(destination), "wb") as writer:
        writer.setnchannels(channels)
        writer.setsampwidth(width)
        writer.setframerate(rate)
        writer.writeframes(frames)
    return True
