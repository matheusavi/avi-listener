from __future__ import annotations

import json
import re
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from avilistener.transcriber import TranscriptResult


class TranscriptWriter:
    def __init__(self, output_dir: str | Path, timezone: str = "America/Sao_Paulo") -> None:
        self.output_dir = Path(output_dir)
        self.timezone = ZoneInfo(timezone)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.output_dir / "events.jsonl"
        self.combined_path = self.output_dir / "combined.txt"

    def write(self, result: TranscriptResult) -> None:
        started = self._format_time(result.started_at)
        ended = self._format_time(result.ended_at)
        line = f"[{started} - {ended}] {result.text}\n"
        source_line = f"[{started} - {ended}] {result.source:<24} | {result.text}\n"

        source_path = self.output_dir / f"{_safe_filename(result.source)}.txt"
        with source_path.open("a", encoding="utf-8") as f:
            f.write(line)
        with self.combined_path.open("a", encoding="utf-8") as f:
            f.write(source_line)
        with self.events_path.open("a", encoding="utf-8") as f:
            event = asdict(result)
            event["started_local"] = started
            event["ended_local"] = ended
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

    def _format_time(self, timestamp: float) -> str:
        return datetime.fromtimestamp(timestamp, self.timezone).strftime("%Y-%m-%d %H:%M:%S")


def _safe_filename(value: str) -> str:
    safe = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", value).strip(" ._")
    return safe[:120] or "unknown"


def clear_transcript_outputs(output_dir: Path) -> None:
    """Empty a transcript directory before writing it again.

    Transcripts are appended to, so re-running a step over a directory that
    still holds the previous run would interleave the two.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in output_dir.glob("*.txt"):
        path.unlink()
    events = output_dir / "events.jsonl"
    if events.exists():
        events.unlink()
