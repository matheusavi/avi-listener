from __future__ import annotations

import wave
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from avilistener.file_transcriber import (
    resample_linear,
    sort_key_for_discord_wav,
    source_name_from_discord_wav,
    timestamp_from_discord_wav,
    transcribe_wav_directory,
    wav_to_audio_chunk,
)
from avilistener.transcriber import TranscriptResult
from avilistener.writer import TranscriptWriter


class FakeTranscriber:
    """Stands in for Transcriber without loading a Whisper model."""

    def __init__(self) -> None:
        self.chunks = []

    def transcribe(self, chunk) -> TranscriptResult | None:
        self.chunks.append(chunk)
        if not chunk.audio.size:
            return None
        return TranscriptResult(
            source=chunk.source,
            text=f"text from {chunk.source}",
            started_at=chunk.started_at,
            ended_at=chunk.ended_at,
            rms=chunk.rms,
        )


def write_wav(path: Path, seconds: float = 0.5, sample_rate: int = 48000, channels: int = 2, frequency: float = 440.0) -> None:
    frames = int(seconds * sample_rate)
    t = np.arange(frames) / sample_rate
    mono = (np.sin(2 * np.pi * frequency * t) * 0.4 * 32767).astype(np.int16)
    data = np.repeat(mono, channels) if channels > 1 else mono
    with wave.open(str(path), "wb") as f:
        f.setnchannels(channels)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(data.tobytes())


class TestFilenameParsing:
    def test_source_name_from_standard_capture_file(self) -> None:
        path = Path("2026-06-09T00-14-39-579Z-Alice-100000000000000001.wav")
        assert source_name_from_discord_wav(path) == "Alice"

    def test_source_name_keeps_dashes_inside_display_name(self) -> None:
        path = Path("2026-06-09T00-15-55-165Z--bob-42-100000000000000002.wav")
        assert source_name_from_discord_wav(path) == "-bob-42"

    def test_source_name_falls_back_to_stem_for_unknown_format(self) -> None:
        assert source_name_from_discord_wav(Path("random-file.wav")) == "random-file"

    def test_timestamp_parsed_as_utc(self) -> None:
        path = Path("2026-06-09T00-14-39-579Z-Alice-100000000000000001.wav")
        value = timestamp_from_discord_wav(path)
        expected = datetime(2026, 6, 9, 0, 14, 39, 579000, tzinfo=timezone.utc).timestamp()
        assert value == pytest.approx(expected)

    def test_timestamp_missing_for_unknown_format(self) -> None:
        assert timestamp_from_discord_wav(Path("random-file.wav")) is None

    def test_sort_key_orders_by_embedded_timestamp(self, tmp_path: Path) -> None:
        older = tmp_path / "2026-06-09T00-01-00-000Z-Bob-2.wav"
        newer = tmp_path / "2026-06-09T00-02-00-000Z-Alice-1.wav"
        for path in (newer, older):
            path.write_bytes(b"")
        ordered = sorted([newer, older], key=sort_key_for_discord_wav)
        assert ordered == [older, newer]


class TestResample:
    def test_identity_when_rates_match(self) -> None:
        audio = np.linspace(-1, 1, 100, dtype=np.float32)
        out = resample_linear(audio, 16000, 16000)
        assert np.array_equal(out, audio)

    def test_downsample_48k_to_16k_length(self) -> None:
        audio = np.zeros(48000, dtype=np.float32)
        out = resample_linear(audio, 48000, 16000)
        assert out.shape == (16000,)
        assert out.dtype == np.float32

    def test_empty_input(self) -> None:
        out = resample_linear(np.zeros(0, dtype=np.float32), 48000, 16000)
        assert out.size == 0


class TestWavToAudioChunk:
    def test_stereo_48k_becomes_mono_16k(self, tmp_path: Path) -> None:
        path = tmp_path / "2026-06-09T00-14-39-579Z-Alice-100000000000000001.wav"
        write_wav(path, seconds=1.0, sample_rate=48000, channels=2)
        chunk = wav_to_audio_chunk("Alice", path)
        assert chunk.sample_rate == 16000
        assert chunk.audio.ndim == 1
        assert chunk.audio.shape[0] == pytest.approx(16000, abs=10)
        assert chunk.rms > 0.1

    def test_timestamps_come_from_filename(self, tmp_path: Path) -> None:
        path = tmp_path / "2026-06-09T00-14-39-579Z-Alice-100000000000000001.wav"
        write_wav(path, seconds=0.5)
        chunk = wav_to_audio_chunk("Alice", path)
        expected_start = datetime(2026, 6, 9, 0, 14, 39, 579000, tzinfo=timezone.utc).timestamp()
        assert chunk.started_at == pytest.approx(expected_start)
        assert chunk.ended_at - chunk.started_at == pytest.approx(0.5, abs=0.01)


class TestTranscribeWavDirectory:
    def make_writer(self, tmp_path: Path) -> TranscriptWriter:
        return TranscriptWriter(output_dir=tmp_path / "out")

    def test_transcribes_and_moves_processed(self, tmp_path: Path) -> None:
        input_dir = tmp_path / "audio"
        input_dir.mkdir()
        a = input_dir / "2026-06-09T00-01-00-000Z-Alice-1.wav"
        b = input_dir / "2026-06-09T00-02-00-000Z-Bob-2.wav"
        write_wav(a)
        write_wav(b)

        transcriber = FakeTranscriber()
        count = transcribe_wav_directory(
            input_dir=input_dir,
            transcriber=transcriber,
            writer=self.make_writer(tmp_path),
            move_processed=True,
        )

        assert count == 2
        assert not a.exists() and not b.exists()
        processed = sorted(p.name for p in (input_dir / "processed").glob("*.wav"))
        assert processed == [a.name, b.name]
        assert [chunk.source for chunk in transcriber.chunks] == ["Alice", "Bob"]

    def test_keep_audio_leaves_files_in_place(self, tmp_path: Path) -> None:
        input_dir = tmp_path / "audio"
        input_dir.mkdir()
        a = input_dir / "2026-06-09T00-01-00-000Z-Alice-1.wav"
        write_wav(a)

        count = transcribe_wav_directory(
            input_dir=input_dir,
            transcriber=FakeTranscriber(),
            writer=self.make_writer(tmp_path),
            move_processed=False,
        )

        assert count == 1
        assert a.exists()
        assert not (input_dir / "processed").exists()

    def test_missing_input_dir_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            transcribe_wav_directory(
                input_dir=tmp_path / "missing",
                transcriber=FakeTranscriber(),
                writer=self.make_writer(tmp_path),
            )

    def test_writer_outputs_per_source_and_combined(self, tmp_path: Path) -> None:
        input_dir = tmp_path / "audio"
        input_dir.mkdir()
        write_wav(input_dir / "2026-06-09T00-01-00-000Z-Alice-1.wav")

        out_dir = tmp_path / "out"
        transcribe_wav_directory(
            input_dir=input_dir,
            transcriber=FakeTranscriber(),
            writer=TranscriptWriter(output_dir=out_dir),
            move_processed=False,
        )

        assert "text from Alice" in (out_dir / "Alice.txt").read_text(encoding="utf-8")
        assert "text from Alice" in (out_dir / "combined.txt").read_text(encoding="utf-8")
        assert (out_dir / "events.jsonl").exists()
