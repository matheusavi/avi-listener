"""Merge a diarized loopback with the microphone transcript into one timeline.

Recording both sides of a call produces two transcripts on different clocks:

- the microphone, transcribed per utterance, with absolute timestamps recovered
  from each segment's filename;
- the loopback, diarized as a whole, with timestamps relative to the start of
  the continuous WAV.

The bridge is the continuous file's own name, which carries the session start.
That is why the recorder puts a UTC timestamp there.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from avilistener.recorder import source_name_from_continuous_wav, timestamp_from_continuous_wav
from avilistener.transcriber import TranscriptResult


@dataclass(frozen=True)
class DiarizedSource:
    """Where a diarization run came from, recovered from its own output."""

    rttm_path: Path
    source_name: str
    started_at: float


def find_diarized_source(diarized_dir: Path) -> DiarizedSource:
    """Work out which recording a diarization directory describes.

    Diarization names its RTTM after the audio it was given, so the session
    start and the source name can both be read back without asking the user to
    repeat the path they already passed.
    """
    from avilistener.timeline import load_timeline

    timeline = load_timeline(diarized_dir)
    if timeline:
        first = timeline["parts"][0]
        return DiarizedSource(diarized_dir / "session-audio.rttm", first["source"], first["started_at"])
    candidates = [
        path
        for path in sorted(diarized_dir.glob("*.rttm"))
        if timestamp_from_continuous_wav(path) is not None
    ]
    if not candidates:
        raise SystemExit(
            f"No continuous-recording RTTM found in {diarized_dir}. "
            "Split a session recording by speaker first."
        )
    if len(candidates) > 1:
        names = ", ".join(path.name for path in candidates)
        raise SystemExit(f"{diarized_dir} describes more than one recording ({names}). Use a fresh --output directory.")

    rttm_path = candidates[0]
    started_at = timestamp_from_continuous_wav(rttm_path)
    source_name = source_name_from_continuous_wav(rttm_path) or "system"
    return DiarizedSource(rttm_path=rttm_path, source_name=source_name, started_at=started_at)


def load_diarized_events(
    diarized_dir: Path,
    started_at: float,
    names: dict[str, str] | None = None,
) -> list[TranscriptResult]:
    """Diarized lines, shifted from relative seconds onto the absolute clock.

    `names` maps diarization labels to people. Applying it here rather than
    re-running diarization means renaming a speaker costs nothing: the
    clustering result is unchanged, only how it is presented.
    """
    events_path = diarized_dir / "events.json"
    if not events_path.exists():
        raise SystemExit(f"No events.json in {diarized_dir}. Split by speaker first.")

    events = json.loads(events_path.read_text(encoding="utf-8"))
    results = []
    for event in events:
        text = (event.get("text") or "").strip()
        if not text:
            continue
        label = str(event.get("speaker_label") or event.get("speaker") or "speaker")
        results.append(
            TranscriptResult(
                source=(names or {}).get(label, str(event.get("speaker") or label)),
                text=text,
                started_at=float(event.get("started_at", started_at + float(event.get("start", 0.0)))),
                ended_at=float(event.get("ended_at", started_at + float(event.get("end", 0.0)))),
                rms=0.0,
            )
        )
    return results


def load_transcript_events(
    events_path: Path,
    exclude_source: str | None,
    rename: dict[str, str] | None = None,
    coverage: list[dict] | None = None,
) -> list[TranscriptResult]:
    """Transcribed segments, minus the source that was diarized.

    The loopback's own segments describe the same audio the diarized lines
    already cover, so including them would print every remote utterance twice.
    """
    if not events_path.exists():
        raise SystemExit(f"No events.jsonl at {events_path}. Transcribe the recording first.")

    results = []
    for line in events_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        event = json.loads(line)
        source = str(event.get("source") or "unknown")
        if coverage is not None:
            # A partial/older diarization cannot erase every clip of its source.
            # Keep boundary clips when their full interval is not covered.
            covered = any(
                source == part["source"]
                and float(event.get("started_at", 0.0)) >= part["started_at"] - 0.5
                and float(event.get("ended_at", 0.0)) <= part["ended_at"] + 0.5
                for part in coverage
            )
            if covered:
                continue
        elif exclude_source is not None and source == exclude_source:
            continue
        text = (event.get("text") or "").strip()
        if not text:
            continue
        results.append(
            TranscriptResult(
                # "mic" names a device, not a person. In a transcript read by
                # humans it should carry whoever was holding it.
                source=(rename or {}).get(source, source),
                text=text,
                started_at=float(event.get("started_at", 0.0)),
                ended_at=float(event.get("ended_at", 0.0)),
                rms=float(event.get("rms", 0.0)),
            )
        )
    return results


def merge_results(*groups: list[TranscriptResult]) -> list[TranscriptResult]:
    """One chronological timeline.

    Overlapping speech stays overlapping: when two people talk at once both
    lines appear, which is what actually happened.
    """
    merged = [result for group in groups for result in group]
    return sorted(merged, key=lambda result: (result.started_at, result.ended_at, result.source))
