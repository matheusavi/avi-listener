"""Boosting quiet audio so NeMo's VAD can hear it.

Whisper normalises internally, so it transcribes quiet audio fine. NeMo's
`vad_multilingual_marblenet` judges absolute level, so a quiet recording yields
no speaker turns and every transcribed line falls back to SPEAKER_UNKNOWN.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest

from avilistener.meeting import diarization_gain, prepare_for_diarization
from avilistener.recorder import write_wav

# Measured from a real recording that diarized to SPEAKER_UNKNOWN: peak 0.0512,
# i.e. -25.8 dB below full scale. NeMo marked 4s of speech in a 34s file.
QUIET_PEAK = 0.05


def tone(seconds: float, amplitude: float, sample_rate: int = 16000) -> np.ndarray:
    t = np.arange(int(seconds * sample_rate)) / sample_rate
    return (np.sin(2 * np.pi * 220 * t) * amplitude).astype(np.float32)


class TestDiarizationGain:
    def test_quiet_recording_is_boosted_towards_full_scale(self) -> None:
        gain = diarization_gain(tone(1.0, QUIET_PEAK))
        assert gain > 5, "a -26 dB recording must be boosted substantially"
        assert QUIET_PEAK * gain <= 1.0, "boosting must not push audio past full scale"

    def test_audio_that_is_already_loud_is_left_alone(self) -> None:
        assert diarization_gain(tone(1.0, 0.9)) == pytest.approx(1.0)

    def test_silence_is_not_amplified(self) -> None:
        assert diarization_gain(np.zeros(16000, dtype=np.float32)) == pytest.approx(1.0)

    def test_gain_is_capped_so_near_silence_does_not_become_noise(self) -> None:
        assert diarization_gain(tone(1.0, 1e-5), max_gain=60.0) == pytest.approx(60.0)

    def test_a_single_click_cannot_suppress_the_gain(self) -> None:
        """Dropped samples produce clicks; a peak-based gain would be defeated."""
        audio = tone(2.0, QUIET_PEAK)
        clean_gain = diarization_gain(audio)
        audio[5000] = 1.0  # one full-scale sample
        assert diarization_gain(audio) == pytest.approx(clean_gain, rel=0.2)

    def test_empty_input(self) -> None:
        assert diarization_gain(np.zeros(0, dtype=np.float32)) == pytest.approx(1.0)


class TestPrepareForDiarization:
    def test_quiet_file_produces_a_louder_copy_and_keeps_the_original(self, tmp_path: Path) -> None:
        source = tmp_path / "quiet.wav"
        write_wav(source, tone(2.0, QUIET_PEAK), 16000)
        before = source.read_bytes()

        prepared = prepare_for_diarization(source, tmp_path)
        assert prepared != source
        assert source.read_bytes() == before, "the archived recording must not be modified"

        with wave.open(str(prepared), "rb") as handle:
            assert handle.getframerate() == 16000
            audio = np.frombuffer(handle.readframes(handle.getnframes()), dtype=np.int16).astype(np.float32) / 32768
        assert float(np.abs(audio).max()) > 0.5, "prepared audio must be loud enough for VAD"

    def test_duration_is_unchanged_so_timestamps_still_line_up(self, tmp_path: Path) -> None:
        source = tmp_path / "quiet.wav"
        write_wav(source, tone(3.0, QUIET_PEAK), 16000)
        prepared = prepare_for_diarization(source, tmp_path)

        with wave.open(str(prepared), "rb") as handle:
            frames = handle.getnframes()
        assert frames == 48000, "a shifted timeline would mislabel every speaker turn"

    def test_loud_file_is_passed_through_untouched(self, tmp_path: Path) -> None:
        source = tmp_path / "loud.wav"
        write_wav(source, tone(1.0, 0.9), 16000)
        assert prepare_for_diarization(source, tmp_path) == source

    def test_unreadable_audio_falls_back_to_the_original(self, tmp_path: Path) -> None:
        source = tmp_path / "weird.m4a"
        source.write_bytes(b"not a wav file at all")
        # NeMo's own loader handles formats we cannot read; never crash here.
        assert prepare_for_diarization(source, tmp_path) == source

    def test_stereo_input_is_downmixed(self, tmp_path: Path) -> None:
        source = tmp_path / "stereo.wav"
        mono = tone(1.0, QUIET_PEAK)
        stereo = np.repeat(mono.reshape(-1, 1), 2, axis=1).reshape(-1)
        with wave.open(str(source), "wb") as handle:
            handle.setnchannels(2)
            handle.setsampwidth(2)
            handle.setframerate(16000)
            handle.writeframes((np.clip(stereo, -1, 1) * 32767).astype(np.int16).tobytes())

        prepared = prepare_for_diarization(source, tmp_path)
        with wave.open(str(prepared), "rb") as handle:
            assert handle.getnchannels() == 1
            assert handle.getnframes() == 16000
