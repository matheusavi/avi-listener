from __future__ import annotations

import threading
import wave
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from avilistener.audio import SourceConfig
from avilistener.file_transcriber import source_name_from_discord_wav, timestamp_from_discord_wav
from avilistener.recorder import (
    AdaptiveGate,
    BrowserStreamRecorder,
    ContinuousWriter,
    continuous_filename,
    SegmentSettings,
    SilenceSegmenter,
    numeric_id_for_name,
    safe_name,
    segment_filename,
    write_wav,
)

SETTINGS = SegmentSettings(
    sample_rate=16000,
    silence_rms_threshold=0.01,
    silence_duration_ms=300,
    preroll_ms=100,
    min_segment_ms=100,
)


def tone(ms: float, amplitude: float, sample_rate: int = 16000) -> np.ndarray:
    frames = int(sample_rate * ms / 1000)
    t = np.arange(frames) / sample_rate
    return (np.sin(2 * np.pi * 440 * t) * amplitude).astype(np.float32)


def collect(settings: SegmentSettings = SETTINGS):
    segments: list[tuple[float, np.ndarray]] = []
    return segments, SilenceSegmenter(settings, lambda started, audio: segments.append((started, audio)))


class TestSilenceSegmenter:
    def test_emits_one_segment_for_speech_then_silence(self) -> None:
        segments, segmenter = collect()
        now = 1000.0
        for _ in range(8):
            now += 0.05
            segmenter.push(tone(50, 0.5), now)
        for _ in range(10):
            now += 0.05
            segmenter.push(tone(50, 0.0), now)

        assert len(segments) == 1
        started, audio = segments[0]
        assert audio.size / SETTINGS.sample_rate >= 0.4
        assert started <= 1000.4

    def test_drops_segments_shorter_than_minimum(self) -> None:
        settings = SegmentSettings(**{**SETTINGS.__dict__, "min_segment_ms": 5000})
        segments, segmenter = collect(settings)
        now = 0.0
        for _ in range(2):
            now += 0.05
            segmenter.push(tone(50, 0.5), now)
        for _ in range(10):
            now += 0.05
            segmenter.push(tone(50, 0.0), now)
        segmenter.finish(now)
        assert segments == []

    def test_finish_flushes_an_open_segment(self) -> None:
        segments, segmenter = collect()
        segmenter.push(tone(200, 0.5), 0.2)
        assert segmenter.is_open
        segmenter.finish(0.2)
        assert len(segments) == 1
        assert not segmenter.is_open

    def test_keeps_preroll_before_speech_so_first_syllable_survives(self) -> None:
        segments, segmenter = collect()
        now = 0.0
        for _ in range(10):
            now += 0.05
            segmenter.push(tone(50, 0.0), now)
        now += 0.05
        segmenter.push(tone(50, 0.5), now)
        segmenter.finish(now)

        assert len(segments) == 1
        duration_ms = segments[0][1].size / SETTINGS.sample_rate * 1000
        # 50ms of speech plus up to preroll_ms (100ms) of buffered silence.
        assert 50 < duration_ms <= 160

    def test_ignores_empty_blocks(self) -> None:
        segments, segmenter = collect()
        segmenter.push(np.zeros(0, dtype=np.float32), 1.0)
        assert segments == []
        assert not segmenter.is_open

    def test_separate_utterances_produce_separate_segments(self) -> None:
        segments, segmenter = collect()
        now = 0.0
        for _ in range(2):
            for _ in range(6):
                now += 0.05
                segmenter.push(tone(50, 0.5), now)
            for _ in range(10):
                now += 0.05
                segmenter.push(tone(50, 0.0), now)
        assert len(segments) == 2


class TestAdaptiveGate:
    """Levels here are taken from real recordings on a quiet microphone.

    Speech measured ~0.004-0.008 RMS against a ~0.0002 noise floor, while the
    speaker loopback ran ~0.04 against ~0.00003. A single fixed threshold of
    0.004 captured the loopback fine but forced shouting into the microphone.
    """

    MIC_FLOOR = 0.0002
    MIC_SPEECH = 0.005
    SYSTEM_FLOOR = 0.00003
    SYSTEM_SPEECH = 0.04

    def gate(self, **overrides) -> AdaptiveGate:
        return AdaptiveGate(SegmentSettings(**overrides))

    def test_quiet_mic_speech_clears_the_gate(self) -> None:
        gate = self.gate()
        for _ in range(60):
            gate.update(self.MIC_FLOOR)
        assert gate.threshold < self.MIC_SPEECH, "normal speech must not need shouting"

    def test_loud_system_audio_still_gates_out_its_noise_floor(self) -> None:
        gate = self.gate()
        for _ in range(60):
            gate.update(self.SYSTEM_FLOOR)
        assert gate.threshold > self.SYSTEM_FLOOR
        assert gate.threshold < self.SYSTEM_SPEECH

    def test_threshold_never_falls_below_the_minimum(self) -> None:
        gate = self.gate(min_threshold=0.001)
        for _ in range(60):
            gate.update(0.0)
        assert gate.threshold == pytest.approx(0.001)

    def test_continuous_loud_audio_cannot_ratchet_the_gate_shut(self) -> None:
        gate = self.gate(max_threshold=0.02)
        for _ in range(300):
            gate.update(0.5)
        assert gate.threshold <= 0.02

    def test_gate_sits_between_floor_and_speech_for_a_realistic_mix(self) -> None:
        gate = self.gate()
        for i in range(200):
            gate.update(self.MIC_SPEECH if i % 4 == 0 else self.MIC_FLOOR)
        assert self.MIC_FLOOR < gate.threshold < self.MIC_SPEECH


class TestAdaptiveSegmentation:
    def test_captures_quiet_speech_that_a_fixed_threshold_would_miss(self) -> None:
        """Regression: speech at 0.003 RMS was silently dropped by the old gate."""
        quiet = SegmentSettings(
            sample_rate=16000, silence_duration_ms=300, preroll_ms=100, min_segment_ms=100
        )
        fixed = SegmentSettings(
            sample_rate=16000, silence_rms_threshold=0.004,
            silence_duration_ms=300, preroll_ms=100, min_segment_ms=100,
        )

        def run(settings: SegmentSettings) -> int:
            segments, segmenter = collect(settings)
            now = 0.0
            for _ in range(20):  # establish a quiet noise floor
                now += 0.05
                segmenter.push(tone(50, 0.0004), now)
            for _ in range(10):  # quiet speech
                now += 0.05
                segmenter.push(tone(50, 0.0045), now)
            for _ in range(10):
                now += 0.05
                segmenter.push(tone(50, 0.0004), now)
            segmenter.finish(now)
            return len(segments)

        assert run(fixed) == 0, "test is meaningless unless the fixed gate misses it"
        assert run(quiet) == 1, "adaptive gate must capture quiet speech"

    def test_explicit_threshold_still_overrides_the_adaptive_gate(self) -> None:
        settings = SegmentSettings(
            sample_rate=16000, silence_rms_threshold=0.5,
            silence_duration_ms=300, preroll_ms=100, min_segment_ms=100,
        )
        segments, segmenter = collect(settings)
        now = 0.0
        for _ in range(10):
            now += 0.05
            segmenter.push(tone(50, 0.1), now)
        segmenter.finish(now)
        assert segments == []


class TestFilenameContract:
    """The recorder's output must be readable by the existing transcriber."""

    def test_filename_round_trips_through_the_transcriber_parser(self) -> None:
        started = datetime(2026, 6, 9, 12, 30, 45, 123000, tzinfo=timezone.utc).timestamp()
        name = segment_filename(started, "mic")

        assert source_name_from_discord_wav(Path(name)) == "mic"
        assert timestamp_from_discord_wav(Path(name)) == pytest.approx(started, abs=0.001)

    def test_source_names_stay_distinct(self) -> None:
        assert numeric_id_for_name("mic") != numeric_id_for_name("system")
        assert numeric_id_for_name("mic") == numeric_id_for_name("mic")
        assert numeric_id_for_name("mic").isdigit()

    def test_safe_name_strips_characters_invalid_on_windows(self) -> None:
        assert safe_name('a<b>:"c/d\\e|f?g*h') == "a_b___c_d_e_f_g_h"
        assert safe_name("   ") == "unknown"

    def test_names_with_spaces_still_parse_back(self) -> None:
        started = datetime(2026, 6, 9, 12, 0, 0, tzinfo=timezone.utc).timestamp()
        name = segment_filename(started, "Meeting Room")
        assert source_name_from_discord_wav(Path(name)) == "Meeting Room"


class TestContinuousWriter:
    """Unbroken session audio, which is what diarization needs."""

    def test_writes_everything_including_silence(self, tmp_path: Path) -> None:
        writer = ContinuousWriter(tmp_path / "c.wav", 16000)
        writer.write(tone(500, 0.5))
        writer.write(tone(500, 0.0))  # silence must be kept, not gated away
        writer.write(tone(500, 0.5))
        writer.close()

        with wave.open(str(tmp_path / "c.wav"), "rb") as handle:
            assert handle.getframerate() == 16000
            assert handle.getnchannels() == 1
            frames = handle.getnframes()
        assert frames == 24000  # 1.5s at 16kHz, gaps intact
        assert writer.seconds == pytest.approx(1.5)

    def test_close_finalises_the_header(self, tmp_path: Path) -> None:
        path = tmp_path / "c.wav"
        writer = ContinuousWriter(path, 16000)
        writer.write(tone(1000, 0.4))
        writer.close()

        # A header left unfinalised reports zero frames and the audio is lost.
        with wave.open(str(path), "rb") as handle:
            assert handle.getnframes() == 16000

    def test_file_size_grows_on_disk_before_close(self, tmp_path: Path) -> None:
        """Live monitoring must not wait for Python's file buffer to flush."""
        path = tmp_path / "c.wav"
        writer = ContinuousWriter(path, 16000)

        writer.write(tone(250, 0.2))
        first_size = path.stat().st_size
        writer.write(tone(250, 0.2))
        second_size = path.stat().st_size

        assert first_size == 44 + 4000 * 2
        assert second_size == 44 + 8000 * 2
        writer.close()

    def test_no_file_is_created_when_nothing_is_recorded(self, tmp_path: Path) -> None:
        writer = ContinuousWriter(tmp_path / "nested" / "c.wav", 16000)
        writer.close()
        assert not (tmp_path / "nested" / "c.wav").exists()

    def test_ignores_empty_blocks(self, tmp_path: Path) -> None:
        writer = ContinuousWriter(tmp_path / "c.wav", 16000)
        writer.write(np.zeros(0, dtype=np.float32))
        writer.write(tone(250, 0.3))
        writer.close()
        assert writer.frames_written == 4000

    def test_transcriber_must_not_treat_a_session_file_as_one_utterance(self, tmp_path: Path) -> None:
        """The continuous file lives apart from the segments on purpose.

        Transcription globs *.wav non-recursively; if a session recording sat
        beside the segments it would be transcribed as a single enormous
        chunk.
        """
        segments_dir = tmp_path / "recordings"
        segments_dir.mkdir()
        write_wav(segments_dir / segment_filename(1_800_000_000.0, "mic"), tone(300, 0.4), 16000)

        continuous_dir = segments_dir / "continuous"
        continuous_dir.mkdir()
        name = continuous_filename(1_800_000_000.0, "mic")
        write_wav(continuous_dir / name, tone(300, 0.4), 16000)

        found = sorted(p.name for p in segments_dir.glob("*.wav"))
        assert len(found) == 1 and found[0].endswith("-mic-1001884403.wav")
        # Even by name alone it is not a segment.
        assert source_name_from_discord_wav(Path(name)) != "mic"


class TestBrowserStreamRecorder:
    def recorder(self, tmp_path: Path) -> BrowserStreamRecorder:
        recorder = BrowserStreamRecorder(
            SourceConfig(name="chrome", kind="browser"),
            tmp_path,
            SETTINGS,
            threading.Event(),
        )
        recorder.start()
        return recorder

    def test_writes_tab_audio_to_continuous_wav_and_speech_clip(self, tmp_path: Path) -> None:
        recorder = self.recorder(tmp_path)
        recorder.push_pcm(tone(500, 0.4).astype("<f4").tobytes(), 16000)
        recorder.push_pcm(tone(500, 0.0).astype("<f4").tobytes(), 16000)
        recorder.join()

        assert recorder.saved == 1
        assert recorder.continuous_path is not None
        with wave.open(str(recorder.continuous_path), "rb") as handle:
            assert handle.getframerate() == 16000
            assert handle.getnframes() == 16000
        [clip] = list(tmp_path.glob("*.wav"))
        assert source_name_from_discord_wav(clip) == "chrome"

    def test_uses_the_browser_context_actual_sample_rate(self, tmp_path: Path) -> None:
        recorder = self.recorder(tmp_path)
        recorder.push_pcm(tone(250, 0.4, sample_rate=48000).astype("<f4").tobytes(), 48000)
        recorder.join()

        assert recorder.continuous_path is not None
        with wave.open(str(recorder.continuous_path), "rb") as handle:
            assert handle.getframerate() == 48000
            assert handle.getnframes() == 12000

    def test_rejects_a_sample_rate_change_mid_recording(self, tmp_path: Path) -> None:
        recorder = self.recorder(tmp_path)
        recorder.push_pcm(tone(100, 0.2).astype("<f4").tobytes(), 16000)
        with pytest.raises(ValueError, match="sample rate changed"):
            recorder.push_pcm(tone(100, 0.2, sample_rate=48000).astype("<f4").tobytes(), 48000)
        recorder.join()

    def test_no_audio_leaves_no_empty_session_file(self, tmp_path: Path) -> None:
        recorder = self.recorder(tmp_path)
        recorder.join()
        assert recorder.continuous_path is None
        assert not list(tmp_path.rglob("*.wav"))


class TestWriteWav:
    def test_writes_mono_pcm_at_the_requested_rate(self, tmp_path: Path) -> None:
        path = tmp_path / "out.wav"
        write_wav(path, tone(500, 0.5), 16000)

        with wave.open(str(path), "rb") as handle:
            assert handle.getnchannels() == 1
            assert handle.getsampwidth() == 2
            assert handle.getframerate() == 16000
            frames = handle.readframes(handle.getnframes())

        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768
        assert audio.size == 8000
        assert 0.4 < float(np.abs(audio).max()) <= 0.51

    def test_clips_out_of_range_samples_instead_of_wrapping(self, tmp_path: Path) -> None:
        path = tmp_path / "loud.wav"
        write_wav(path, np.array([2.0, -2.0, 0.0], dtype=np.float32), 16000)

        with wave.open(str(path), "rb") as handle:
            audio = np.frombuffer(handle.readframes(handle.getnframes()), dtype=np.int16)

        # Wrapping would flip the sign and produce a loud click.
        assert audio[0] > 32000
        assert audio[1] < -32000
