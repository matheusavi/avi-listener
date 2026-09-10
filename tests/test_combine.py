"""Merging a diarized loopback with the microphone transcript.

The two sides arrive on different clocks: diarization is relative to the start
of the continuous WAV, the microphone is absolute. Getting the offset wrong
does not fail loudly, it just silently interleaves the conversation wrongly.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from avilistener.combine import (
    find_diarized_source,
    load_diarized_events,
    load_transcript_events,
    merge_results,
)
from avilistener.transcriber import TranscriptResult

SESSION_START = datetime(2026, 8, 24, 19, 28, 50, 97000, tzinfo=timezone.utc).timestamp()
RTTM_NAME = "2026-08-24T19-28-50-097Z-system-continuous.rttm"


def write_diarized(directory: Path, events: list[dict], rttm_name: str = RTTM_NAME) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / rttm_name).write_text("SPEAKER x 1 0.0 1.0 <NA> <NA> speaker_0 <NA> <NA>\n", encoding="utf-8")
    (directory / "events.json").write_text(json.dumps(events), encoding="utf-8")


def write_transcript(directory: Path, events: list[dict]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "events.jsonl"
    path.write_text("\n".join(json.dumps(event) for event in events), encoding="utf-8")
    return path


class TestFindDiarizedSource:
    def test_recovers_session_start_and_source_from_the_rttm_name(self, tmp_path: Path) -> None:
        write_diarized(tmp_path, [])
        source = find_diarized_source(tmp_path)
        assert source.source_name == "system"
        assert source.started_at == pytest.approx(SESSION_START)

    def test_rejects_a_directory_with_no_recognisable_recording(self, tmp_path: Path) -> None:
        tmp_path.joinpath("something.rttm").write_text("", encoding="utf-8")
        with pytest.raises(SystemExit):
            find_diarized_source(tmp_path)

    def test_refuses_to_guess_between_two_recordings(self, tmp_path: Path) -> None:
        """A reused output directory must not silently pick the wrong session."""
        write_diarized(tmp_path, [])
        tmp_path.joinpath("2026-08-24T18-11-51-160Z-system-continuous.rttm").write_text("", encoding="utf-8")
        with pytest.raises(SystemExit):
            find_diarized_source(tmp_path)


class TestLoadDiarizedEvents:
    def test_shifts_relative_times_onto_the_absolute_clock(self, tmp_path: Path) -> None:
        write_diarized(tmp_path, [{"start": 7.69, "end": 15.28, "speaker": "speaker_2", "text": "hello"}])
        results = load_diarized_events(tmp_path, SESSION_START)

        assert len(results) == 1
        assert results[0].source == "speaker_2"
        assert results[0].started_at == pytest.approx(SESSION_START + 7.69)
        assert results[0].ended_at == pytest.approx(SESSION_START + 15.28)

    def test_skips_empty_text(self, tmp_path: Path) -> None:
        write_diarized(tmp_path, [{"start": 1.0, "end": 2.0, "speaker": "speaker_0", "text": "   "}])
        assert load_diarized_events(tmp_path, SESSION_START) == []

    def test_missing_events_file_is_a_clear_error(self, tmp_path: Path) -> None:
        tmp_path.joinpath(RTTM_NAME).write_text("", encoding="utf-8")
        with pytest.raises(SystemExit):
            load_diarized_events(tmp_path, SESSION_START)


class TestLoadTranscriptEvents:
    def test_excludes_the_source_that_was_diarized(self, tmp_path: Path) -> None:
        """Otherwise every remote utterance appears twice."""
        path = write_transcript(
            tmp_path,
            [
                {"source": "mic", "text": "mine", "started_at": 10.0, "ended_at": 11.0},
                {"source": "system", "text": "theirs", "started_at": 12.0, "ended_at": 13.0},
            ],
        )
        results = load_transcript_events(path, exclude_source="system")
        assert [r.source for r in results] == ["mic"]

    def test_keeps_everything_when_nothing_is_excluded(self, tmp_path: Path) -> None:
        path = write_transcript(
            tmp_path,
            [
                {"source": "mic", "text": "a", "started_at": 1.0, "ended_at": 2.0},
                {"source": "system", "text": "b", "started_at": 3.0, "ended_at": 4.0},
            ],
        )
        assert len(load_transcript_events(path, exclude_source=None)) == 2

    def test_tolerates_blank_lines(self, tmp_path: Path) -> None:
        tmp_path.mkdir(parents=True, exist_ok=True)
        path = tmp_path / "events.jsonl"
        path.write_text(
            json.dumps({"source": "mic", "text": "a", "started_at": 1.0, "ended_at": 2.0}) + "\n\n",
            encoding="utf-8",
        )
        assert len(load_transcript_events(path, exclude_source=None)) == 1

    def test_missing_file_is_a_clear_error(self, tmp_path: Path) -> None:
        with pytest.raises(SystemExit):
            load_transcript_events(tmp_path / "nope.jsonl", exclude_source=None)


class TestMergeResults:
    def result(self, source: str, start: float) -> TranscriptResult:
        return TranscriptResult(source=source, text=source, started_at=start, ended_at=start + 1, rms=0.0)

    def test_interleaves_both_sides_chronologically(self) -> None:
        mic = [self.result("mic", 100.0), self.result("mic", 108.0)]
        diarized = [self.result("speaker_1", 104.0), self.result("speaker_2", 112.0)]

        merged = merge_results(diarized, mic)
        assert [r.source for r in merged] == ["mic", "speaker_1", "mic", "speaker_2"]

    def test_overlapping_speech_keeps_both_lines(self) -> None:
        """When two people talk at once, both were really said."""
        merged = merge_results([self.result("speaker_1", 100.0)], [self.result("mic", 100.0)])
        assert len(merged) == 2

    def test_empty_inputs(self) -> None:
        assert merge_results([], []) == []
