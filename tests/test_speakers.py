"""Picking a sample of each voice so a speaker can be identified by ear.

Reading someone's lines is a poor way to recognise them; a few seconds of audio
settles it. The sample has to be of that speaker and not of whoever spoke just
before, which is what the boundary trimming is for.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest

from avilistener.meeting import DiarizationTurn
from avilistener.recorder import write_wav
from avilistener.speakers import extract_wav_slice, pick_sample_turn, speaker_labels


def turns(*spans: tuple[float, float, str]) -> list[DiarizationTurn]:
    return [DiarizationTurn(start=s, end=e, speaker=name) for s, e, name in spans]


def tone(seconds: float, sample_rate: int = 16000) -> np.ndarray:
    t = np.arange(int(seconds * sample_rate)) / sample_rate
    return (np.sin(2 * np.pi * 220 * t) * 0.5).astype(np.float32)


class TestSpeakerLabels:
    def test_orders_by_how_much_each_spoke(self) -> None:
        found = speaker_labels(turns((0, 2, "speaker_1"), (2, 12, "speaker_0"), (12, 15, "speaker_1")))
        assert found == ["speaker_0", "speaker_1"]

    def test_no_turns(self) -> None:
        assert speaker_labels([]) == []


class TestPickSampleTurn:
    def test_uses_the_longest_turn(self) -> None:
        """A long turn is the least likely to be a misassigned fragment."""
        span = pick_sample_turn(
            turns((0, 1, "speaker_0"), (10, 40, "speaker_0"), (50, 52, "speaker_0")),
            "speaker_0",
            max_seconds=6.0,
        )
        assert span is not None
        assert 10 <= span[0] < 11, "sample should come from the 30s turn"

    def test_caps_the_sample_length(self) -> None:
        span = pick_sample_turn(turns((0, 300, "speaker_0")), "speaker_0", max_seconds=6.0)
        assert span[1] - span[0] == pytest.approx(6.0)

    def test_trims_the_edges_so_the_previous_speaker_does_not_bleed_in(self) -> None:
        span = pick_sample_turn(turns((10, 20, "speaker_0")), "speaker_0", edge_trim=0.25)
        assert span[0] > 10.0
        assert span[1] < 20.0

    def test_a_turn_shorter_than_the_trim_is_still_usable(self) -> None:
        span = pick_sample_turn(turns((5, 5.2, "speaker_0")), "speaker_0", edge_trim=0.25)
        assert span is not None
        assert span[1] > span[0]

    def test_unknown_speaker(self) -> None:
        assert pick_sample_turn(turns((0, 5, "speaker_0")), "speaker_9") is None


class TestExtractWavSlice:
    def test_extracts_only_the_requested_span(self, tmp_path: Path) -> None:
        source = tmp_path / "session.wav"
        write_wav(source, tone(30.0), 16000)

        assert extract_wav_slice(source, tmp_path / "s.wav", 10.0, 16.0) is True
        with wave.open(str(tmp_path / "s.wav"), "rb") as handle:
            assert handle.getframerate() == 16000
            assert handle.getnframes() == pytest.approx(6 * 16000, abs=2)

    def test_span_past_the_end_is_clamped(self, tmp_path: Path) -> None:
        source = tmp_path / "session.wav"
        write_wav(source, tone(5.0), 16000)

        assert extract_wav_slice(source, tmp_path / "s.wav", 3.0, 99.0) is True
        with wave.open(str(tmp_path / "s.wav"), "rb") as handle:
            assert handle.getnframes() == pytest.approx(2 * 16000, abs=2)

    def test_empty_span_writes_nothing(self, tmp_path: Path) -> None:
        source = tmp_path / "session.wav"
        write_wav(source, tone(5.0), 16000)

        assert extract_wav_slice(source, tmp_path / "s.wav", 4.0, 4.0) is False
        assert not (tmp_path / "s.wav").exists()

    def test_unreadable_source_reports_failure_rather_than_raising(self, tmp_path: Path) -> None:
        broken = tmp_path / "broken.wav"
        broken.write_bytes(b"not a wav")
        assert extract_wav_slice(broken, tmp_path / "s.wav", 0.0, 1.0) is False

    def test_missing_source(self, tmp_path: Path) -> None:
        assert extract_wav_slice(tmp_path / "nope.wav", tmp_path / "s.wav", 0.0, 1.0) is False
