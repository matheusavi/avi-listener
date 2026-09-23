from __future__ import annotations

import threading
import time
import wave
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from avilistener.live import LiveLine, LiveTranscriber, format_live_line
from avilistener.recorder import SegmentSettings, SilenceSegmenter, segment_filename, write_wav
from avilistener.transcriber import TranscriptResult

SETTINGS = SegmentSettings(
    sample_rate=16000,
    silence_rms_threshold=0.01,
    silence_duration_ms=300,
    preroll_ms=100,
    min_segment_ms=100,
)

TIMEOUT = 10.0


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


class BlockingTranscriber:
    """Holds the worker inside `transcribe` until the test releases it."""

    def __init__(self) -> None:
        self.release = threading.Event()
        self.entered = threading.Event()

    def transcribe(self, chunk) -> TranscriptResult | None:
        self.entered.set()
        self.release.wait(TIMEOUT)
        return TranscriptResult(
            source=chunk.source,
            text=f"text from {chunk.source}",
            started_at=chunk.started_at,
            ended_at=chunk.ended_at,
            rms=chunk.rms,
        )


class FailingTranscriber:
    """Raises for clips whose name contains a marker, succeeds otherwise."""

    def __init__(self, fail_on: str) -> None:
        self.fail_on = fail_on
        self.seen: list[str] = []

    def transcribe(self, chunk) -> TranscriptResult | None:
        self.seen.append(chunk.source)
        if chunk.source == self.fail_on:
            raise RuntimeError("model exploded")
        return TranscriptResult(
            source=chunk.source,
            text=f"text from {chunk.source}",
            started_at=chunk.started_at,
            ended_at=chunk.ended_at,
            rms=chunk.rms,
        )


def tone(ms: float, amplitude: float, sample_rate: int = 16000) -> np.ndarray:
    frames = int(sample_rate * ms / 1000)
    t = np.arange(frames) / sample_rate
    return (np.sin(2 * np.pi * 440 * t) * amplitude).astype(np.float32)


def write_clip(
    directory: Path,
    source: str,
    started_at: float = 1_700_000_000.0,
    seconds: float = 0.5,
    amplitude: float = 0.4,
) -> Path:
    """Write a clip exactly as the recorder would, so parsing matches."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / segment_filename(started_at, source)
    write_wav(path, tone(seconds * 1000, amplitude), 16000)
    return path


def write_empty_clip(directory: Path, source: str, started_at: float = 1_700_000_000.0) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / segment_filename(started_at, source)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"")
    return path


def collector() -> tuple[list[LiveLine], list[tuple[Path, Exception]]]:
    return [], []


class TestLiveTranscriber:
    def test_emits_one_line_per_clip_in_submission_order(self, tmp_path: Path) -> None:
        lines, errors = collector()
        live = LiveTranscriber(FakeTranscriber(), lines.append, lambda p, e: errors.append((p, e)))
        live.start()

        first = write_clip(tmp_path, "mic", started_at=1_700_000_000.0)
        second = write_clip(tmp_path, "system", started_at=1_700_000_010.0)
        live.submit("mic", first, 0.5)
        live.submit("system", second, 0.5)
        live.stop(drain=True, timeout=TIMEOUT)

        assert errors == []
        assert [line.source for line in lines] == ["mic", "system"]
        assert [line.text for line in lines] == ["text from mic", "text from system"]
        assert live.lines_emitted == 2
        assert live.clips_skipped == 0
        for line in lines:
            assert line.transcribed_at >= line.ended_at
            assert line.latency >= 0

    def test_clip_the_model_cannot_read_is_skipped_not_emitted(self, tmp_path: Path) -> None:
        lines, errors = collector()
        live = LiveTranscriber(FakeTranscriber(), lines.append, lambda p, e: errors.append((p, e)))
        live.start()

        live.submit("mic", write_empty_clip(tmp_path, "mic"), 0.0)
        live.stop(drain=True, timeout=TIMEOUT)

        assert lines == []
        assert errors == []
        assert live.clips_skipped == 1
        assert live.lines_emitted == 0

    def test_a_failing_clip_reports_and_the_worker_survives(self, tmp_path: Path) -> None:
        lines, errors = collector()
        live = LiveTranscriber(FailingTranscriber("mic"), lines.append, lambda p, e: errors.append((p, e)))
        live.start()

        bad = write_clip(tmp_path, "mic", started_at=1_700_000_000.0)
        good = write_clip(tmp_path, "system", started_at=1_700_000_010.0)
        live.submit("mic", bad, 0.5)
        live.submit("system", good, 0.5)
        live.stop(drain=True, timeout=TIMEOUT)

        assert [path for path, _ in errors] == [bad]
        assert isinstance(errors[0][1], RuntimeError)
        assert [line.source for line in lines] == ["system"]

    def test_submit_never_blocks_the_recorder(self, tmp_path: Path) -> None:
        """The recorder thread calls submit; blocking it would lose audio."""
        lines, errors = collector()
        transcriber = BlockingTranscriber()
        live = LiveTranscriber(transcriber, lines.append, lambda p, e: errors.append((p, e)))
        live.start()

        paths = [
            write_clip(tmp_path, "mic", started_at=1_700_000_000.0 + index * 10)
            for index in range(3)
        ]
        live.submit("mic", paths[0], 0.5)
        assert transcriber.entered.wait(TIMEOUT), "worker never picked up the first clip"

        started = time.monotonic()
        live.submit("mic", paths[1], 0.5)
        live.submit("mic", paths[2], 0.5)
        elapsed = time.monotonic() - started

        assert elapsed < 1.0, "submit waited for the busy worker"
        assert live.pending == 3
        assert lines == []

        transcriber.release.set()
        live.stop(drain=True, timeout=TIMEOUT)
        assert len(lines) == 3
        assert live.pending == 0

    def test_stop_without_drain_abandons_the_queue(self, tmp_path: Path) -> None:
        """Queued clips are dropped, so a stop cannot wait on a long backlog."""
        lines, errors = collector()
        transcriber = BlockingTranscriber()
        live = LiveTranscriber(transcriber, lines.append, lambda p, e: errors.append((p, e)))
        live.start()

        for index in range(3):
            live.submit("mic", write_clip(tmp_path, "mic", started_at=1_700_000_000.0 + index * 10), 0.5)
        assert transcriber.entered.wait(TIMEOUT), "worker never picked up the first clip"

        # Stop from another thread: the worker is held inside the first clip,
        # so the queue is only released once the abandon flag is definitely up.
        stopper = threading.Thread(target=live.stop, kwargs={"drain": False, "timeout": TIMEOUT})
        stopper.start()
        assert live._abandon.wait(TIMEOUT), "stop(drain=False) must raise the abandon flag first"
        transcriber.release.set()
        stopper.join(TIMEOUT)

        assert not stopper.is_alive()
        assert len(lines) == 1, "only the clip already in flight should be transcribed"
        assert live.pending == 0

    def test_latency_summary_tracks_emitted_lines(self, tmp_path: Path) -> None:
        lines, _ = collector()
        live = LiveTranscriber(FakeTranscriber(), lines.append)
        live.start()
        live.submit("mic", write_clip(tmp_path, "mic"), 0.5)
        live.stop(drain=True, timeout=TIMEOUT)

        assert live.max_latency == pytest.approx(lines[0].latency)
        assert live.average_latency == pytest.approx(lines[0].latency)

    def test_stop_is_safe_before_start_and_twice(self, tmp_path: Path) -> None:
        live = LiveTranscriber(FakeTranscriber(), lambda line: None)
        live.stop(drain=True, timeout=TIMEOUT)
        live.start()
        live.submit("mic", write_clip(tmp_path, "mic"), 0.5)
        live.stop(drain=True, timeout=TIMEOUT)
        live.stop(drain=True, timeout=TIMEOUT)
        assert live.lines_emitted == 1


class TestFormatLiveLine:
    def line(self, **overrides) -> LiveLine:
        started_at = datetime(2026, 6, 9, 12, 30, 45, tzinfo=timezone.utc).timestamp()
        defaults = dict(
            source="mic",
            text="hello there",
            started_at=started_at,
            ended_at=started_at + 2.0,
            transcribed_at=started_at + 4.3,
        )
        defaults.update(overrides)
        return LiveLine(**defaults)

    def test_shows_source_latency_and_text(self) -> None:
        rendered = format_live_line(self.line())
        assert "mic (2.3s): hello there" in rendered

    def test_time_is_the_clip_start_in_local_time(self) -> None:
        line = self.line()
        expected = datetime.fromtimestamp(line.started_at).strftime("%H:%M:%S")
        assert rendered_time(format_live_line(line)) == expected

    def test_latency_uses_one_decimal(self) -> None:
        line = self.line(transcribed_at=self.line().ended_at + 0.04)
        assert "(0.0s)" in format_live_line(line)


def rendered_time(rendered: str) -> str:
    return rendered[1 : rendered.index("]")]


class TestRecorderCallbackIntegration:
    """Proves the recorder's save callback and the live path fit together.

    The recorder itself needs a microphone, so this drives its segmenter and
    reproduces `_save` instead - the same filename, the same `on_saved`
    signature - which is where the two halves actually meet.
    """

    def test_an_utterance_becomes_a_live_line(self, tmp_path: Path) -> None:
        lines, errors = collector()
        live = LiveTranscriber(FakeTranscriber(), lines.append, lambda p, e: errors.append((p, e)))
        live.start()

        saved: list[Path] = []

        def save(started_at: float, audio: np.ndarray) -> None:
            path = tmp_path / segment_filename(started_at, "mic")
            write_wav(path, audio, SETTINGS.sample_rate)
            saved.append(path)
            live.submit("mic", path, audio.size / SETTINGS.sample_rate)

        segmenter = SilenceSegmenter(SETTINGS, save)
        now = 1_700_000_000.0
        for _ in range(8):
            now += 0.05
            segmenter.push(tone(50, 0.5), now)
        for _ in range(10):
            now += 0.05
            segmenter.push(tone(50, 0.0), now)
        segmenter.finish(now)

        live.stop(drain=True, timeout=TIMEOUT)

        assert len(saved) == 1, "the segmenter should close exactly one utterance"
        assert saved[0].exists(), "the clip must still be on disk for the offline pipeline"
        assert errors == []
        assert [line.source for line in lines] == ["mic"]
        assert lines[0].text == "text from mic"
        assert lines[0].ended_at > lines[0].started_at
