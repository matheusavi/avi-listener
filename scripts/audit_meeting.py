"""Check a multipart transcript against its originals and save a readable audit."""
from __future__ import annotations

import argparse
import json
import wave
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from avilistener.server.workspace import Workspace
from avilistener.timeline import inspect_parts, sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("workspace", type=Path)
    parser.add_argument("project")
    parser.add_argument("meeting")
    args = parser.parse_args()
    meeting = Workspace(args.workspace).project(args.project).meeting(args.meeting)
    timeline = json.loads((meeting.diarized_dir / "timeline.json").read_text(encoding="utf-8"))
    parts = timeline["parts"]
    assert inspect_parts(meeting.shared_parts()) == parts, "Audio inputs differ from the processed timeline"
    with wave.open(str(meeting.diarized_dir / "session-audio.wav"), "rb") as audio:
        assert audio.getnframes() == sum(round(p["duration"] * 16000) for p in parts)

    events = json.loads((meeting.diarized_dir / "events.json").read_text(encoding="utf-8"))
    merged = [json.loads(line) for line in (meeting.merged_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    source_events = [json.loads(line) for line in (meeting.transcripts_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert events and merged, "Empty output"
    def signature(event):
        return (event["source"], event["text"], round(event["started_at"], 5), round(event["ended_at"], 5))
    available = Counter(signature(event) for event in merged)
    for event in events:
        part = parts[event["part"]]
        assert part["started_at"] <= event["started_at"] <= event["ended_at"] <= part["ended_at"] + .001
        assert abs(event["started_at"] - (part["started_at"] + event["start"] - part["offset"])) < .001
        expected = dict(event, source=(meeting.config.get("speaker_names") or {}).get(event["speaker_label"], event["speaker"]))
        assert available[signature(expected)] > 0, "A diarized line is absent from the merge"
        available[signature(expected)] -= 1
    retained = 0
    for event in source_events:
        covered = any(event["source"] == p["source"] and event["started_at"] >= p["started_at"] - .5
                      and event["ended_at"] <= p["ended_at"] + .5 for p in parts)
        if covered:
            continue
        expected = dict(event)
        if expected["source"] == "mic":
            expected["source"] = meeting.config.get("me") or "host"
        assert available[signature(expected)] > 0, "A microphone/uncovered line is absent from the merge"
        available[signature(expected)] -= 1
        retained += 1
    assert not any(available.values()), "Unexpected duplicate or extra lines in the merge"
    assert [e["started_at"] for e in merged] == sorted(e["started_at"] for e in merged)
    for source in json.loads((meeting.transcripts_dir / "inputs.json").read_text(encoding="utf-8")):
        path = meeting.recordings_dir / source["filename"]
        if not path.exists():
            path = meeting.recordings_dir / "processed" / source["filename"]
        assert sha256(path) == source["sha256"], "A transcribed clip changed"
    report = {
        "verified_at": datetime.now(timezone.utc).isoformat(), "status": "passed",
        "parts": len(parts), "recorded_seconds": sum(p["duration"] for p in parts),
        "diarized_lines_per_part": dict(Counter(e["part"] for e in events)),
        "retained_microphone_and_uncovered_lines": retained, "merged_lines": len(merged),
        "speakers": dict(Counter(e["source"] for e in merged)),
        "gaps_seconds": [parts[i + 1]["started_at"] - p["ended_at"] for i, p in enumerate(parts[:-1])],
        "combined_sha256": sha256(meeting.merged_dir / "combined.txt"),
    }
    meeting.logs_dir.mkdir(exist_ok=True)
    (meeting.logs_dir / "processing-audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
