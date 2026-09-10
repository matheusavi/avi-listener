"""Record audio sources to WAV files for transcription later.

Recording to disk rather than transcribing live is what makes everything else
possible: the audio can be re-transcribed with a better model, split by voice,
or kept as evidence of what was said.

Files are named exactly like the Discord receiver's output, so transcription
ingests both with no changes and the source name (`mic`, `system`, ...) becomes
the speaker label.

This is the practical way to record a whole video call. Capturing the speaker
loopback picks up every remote participant regardless of what the meeting app
does internally, which browser-side capture cannot always manage.
"""

from __future__ import annotations

import re
import threading
import time
import wave
from collections import deque
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO, Callable

import numpy as np

from avilistener.audio import SourceConfig, resolve_recorder_source


@dataclass(frozen=True)
class SegmentSettings:
    """Silence-splitting parameters, mirroring the Discord receiver.

    `silence_rms_threshold = None` (the default) adapts the gate to each source
    instead of using one fixed number. A quiet microphone and a speaker loopback
    can differ by more than 10x in level, so a single threshold either misses
    normal speech on the quiet source or lets noise through on the loud one.
    """

    sample_rate: int = 16000
    silence_rms_threshold: float | None = None
    silence_duration_ms: int = 1500
    preroll_ms: int = 300
    min_segment_ms: int = 700
    # Adaptive gate: threshold = noise floor * multiplier, clamped.
    noise_multiplier: float = 5.0
    noise_percentile: float = 10.0
    min_threshold: float = 0.0006
    max_threshold: float = 0.02
    noise_window_blocks: int = 240


class AdaptiveGate:
    """Tracks the noise floor and gates speech relative to it.

    Speech is loud compared with a source's own noise floor even when it is
    quiet in absolute terms, so measuring the floor is far more robust than
    asking the user to tune a number per microphone.
    """

    def __init__(self, settings: SegmentSettings) -> None:
        self.settings = settings
        self._recent: deque[float] = deque(maxlen=max(8, settings.noise_window_blocks))
        self.threshold = settings.min_threshold

    def update(self, rms: float) -> float:
        self._recent.append(rms)
        # Percentile of recent blocks approximates the floor: most of a
        # conversation is silence, and clamping stops a continuously loud
        # source from ratcheting its own gate up until speech is cut off.
        floor = float(np.percentile(np.asarray(self._recent, dtype=np.float64), self.settings.noise_percentile))
        self.threshold = float(
            min(max(floor * self.settings.noise_multiplier, self.settings.min_threshold), self.settings.max_threshold)
        )
        return self.threshold


class SilenceSegmenter:
    """Splits a continuous stream into utterances separated by silence.

    Keeping a little audio from before speech starts (`preroll_ms`) matters:
    without it the first syllable is clipped, which Whisper often mistranscribes.
    """

    def __init__(self, settings: SegmentSettings, on_segment: Callable[[float, np.ndarray], None]) -> None:
        self.settings = settings
        self.on_segment = on_segment
        self._gate = AdaptiveGate(settings) if settings.silence_rms_threshold is None else None
        self._preroll: list[np.ndarray] = []
        self._preroll_frames = 0
        self._blocks: list[np.ndarray] | None = None
        self._started_at = 0.0
        self._last_voice_at = 0.0

    @property
    def is_open(self) -> bool:
        return self._blocks is not None

    def push(self, block: np.ndarray, now: float) -> None:
        block = np.asarray(block, dtype=np.float32).reshape(-1)
        if block.size == 0:
            return

        rms = float(np.sqrt(np.mean(np.square(block))))
        threshold = self._gate.update(rms) if self._gate is not None else self.settings.silence_rms_threshold
        is_voice = rms >= threshold
        block_seconds = block.size / self.settings.sample_rate

        if is_voice and self._blocks is None:
            preroll_seconds = self._preroll_frames / self.settings.sample_rate
            self._started_at = now - block_seconds - preroll_seconds
            self._blocks = list(self._preroll)
            self._preroll = []
            self._preroll_frames = 0

        if self._blocks is not None:
            if is_voice:
                self._last_voice_at = now
            self._blocks.append(block)
            silent_ms = (now - self._last_voice_at) * 1000
            if not is_voice and silent_ms >= self.settings.silence_duration_ms:
                self._close(now)
                self._remember_preroll(block)
        else:
            self._remember_preroll(block)

    def finish(self, now: float) -> None:
        if self._blocks is not None:
            self._close(now)

    def _close(self, now: float) -> None:
        blocks = self._blocks or []
        self._blocks = None
        if not blocks:
            return
        audio = np.concatenate(blocks)
        duration_ms = (audio.size / self.settings.sample_rate) * 1000
        if duration_ms < self.settings.min_segment_ms:
            return
        self.on_segment(self._started_at, audio)

    def _remember_preroll(self, block: np.ndarray) -> None:
        max_frames = int(self.settings.sample_rate * (self.settings.preroll_ms / 1000))
        self._preroll.append(block)
        self._preroll_frames += block.size
        while self._preroll_frames > max_frames and len(self._preroll) > 1:
            removed = self._preroll.pop(0)
            self._preroll_frames -= removed.size


def numeric_id_for_name(name: str) -> str:
    """Stable, purely numeric id for a source name.

    The transcriber's filename pattern requires the trailing field to be digits,
    so a name alone cannot be used.
    """
    value = 0
    for char in name:
        value = (value * 131 + ord(char)) % 1_000_000_007
    return str(1_000_000_000 + value)


def segment_filename(started_at: float, source_name: str) -> str:
    """Name a segment the way the transcriber expects to read it."""
    stamp = datetime.fromtimestamp(started_at, timezone.utc).strftime("%Y-%m-%dT%H-%M-%S-%f")[:-3]
    safe = safe_name(source_name)
    return f"{stamp}Z-{safe}-{numeric_id_for_name(safe)}.wav"


def safe_name(value: str) -> str:
    cleaned = "".join("_" if char in '<>:"/\\|?*' or ord(char) < 32 else char for char in value)
    return " ".join(cleaned.split()).strip(" ._")[:80] or "unknown"


def float_to_pcm16(audio: np.ndarray) -> np.ndarray:
    """Clip rather than wrap, so a loud sample cannot flip sign into a click."""
    samples = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
    return (samples * 32767.0).astype(np.int16)


def write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(float_to_pcm16(audio).tobytes())


def continuous_filename(started_at: float, source_name: str) -> str:
    """Name for an unbroken session recording.

    Deliberately does not match the segment pattern the transcriber parses, and
    these files live in a subdirectory, so transcription cannot pick one up and
    try to transcribe an hour as a single utterance.
    """
    stamp = datetime.fromtimestamp(started_at, timezone.utc).strftime("%Y-%m-%dT%H-%M-%S-%f")[:-3]
    return f"{stamp}Z-{safe_name(source_name)}-continuous.wav"


_CONTINUOUS_PATTERN = re.compile(r"^(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}-\d{3})Z-(.+)-continuous$")


def timestamp_from_continuous_wav(path: Path) -> float | None:
    """Session start, as an absolute epoch, recovered from the filename.

    This is what lets diarization results (which are relative to the start of
    the file) be placed on the same clock as the microphone transcript.
    """
    match = _CONTINUOUS_PATTERN.match(path.stem)
    if not match:
        return None
    try:
        value = datetime.strptime(match.group(1), "%Y-%m-%dT%H-%M-%S-%f").replace(tzinfo=timezone.utc)
    except ValueError:
        return None
    return value.timestamp()


def source_name_from_continuous_wav(path: Path) -> str | None:
    match = _CONTINUOUS_PATTERN.match(path.stem)
    return match.group(2) if match else None


class ContinuousWriter:
    """Streams every sample to one unbroken WAV, silence included.

    Diarization needs the real timeline: concatenating the utterance segments
    would remove the gaps between them and misrepresent when people spoke. The
    file is written incrementally so a long meeting never has to fit in memory.
    """

    def __init__(self, path: Path, sample_rate: int) -> None:
        self.path = path
        self.sample_rate = sample_rate
        self.frames_written = 0
        self._raw_handle: BinaryIO | None = None
        self._handle: wave.Wave_write | None = None

    def write(self, block: np.ndarray) -> None:
        block = np.asarray(block, dtype=np.float32).reshape(-1)
        if block.size == 0:
            return
        if self._handle is None:
            # Opened lazily so a source that fails immediately leaves no file.
            # The raw file is deliberately unbuffered: the dashboard monitors
            # its on-disk size while recording, and a buffered handle can keep
            # several blocks invisible until it happens to flush.
            self.path.parent.mkdir(parents=True, exist_ok=True)
            raw_handle = self.path.open("wb", buffering=0)
            try:
                handle = wave.open(raw_handle, "wb")
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(self.sample_rate)
            except Exception:
                raw_handle.close()
                raise
            self._raw_handle = raw_handle
            self._handle = handle
        self._handle.writeframes(float_to_pcm16(block).tobytes())
        self.frames_written += block.size

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()  # rewrites the RIFF header with the final length
            self._handle = None
        if self._raw_handle is not None:
            self._raw_handle.close()
            self._raw_handle = None

    @property
    def seconds(self) -> float:
        return self.frames_written / self.sample_rate


@dataclass
class LevelReport:
    """What a source actually sounds like, in the units the gate uses."""

    source: str
    blocks: int = 0
    noise_floor: float = 0.0
    threshold: float = 0.0
    peak_block: float = 0.0
    voiced_ratio: float = 0.0
    error: Exception | None = None


class LevelProbe(threading.Thread):
    """Measures a source without writing any audio to disk.

    Levels are invisible otherwise, which turns "it misses my voice" into
    guesswork. Nothing is recorded to disk, so this is safe to run to inspect a
    microphone.
    """

    def __init__(self, source: SourceConfig, settings: SegmentSettings, stop_event: threading.Event, duration: float) -> None:
        super().__init__(name=f"levels-{source.name}", daemon=True)
        self.source = source
        self.settings = settings
        self.stop_event = stop_event
        self.duration = duration
        self.report = LevelReport(source=source.name)

    def run(self) -> None:
        try:
            self._probe()
        except Exception as exc:
            self.report.error = exc

    def _probe(self) -> None:
        gate = AdaptiveGate(self.settings)
        recorder_source = resolve_recorder_source(self.source)
        block_frames = max(256, int(self.settings.sample_rate * 0.25))
        levels: list[float] = []
        voiced = 0
        deadline = time.monotonic() + self.duration

        with recorder_source.recorder(samplerate=self.settings.sample_rate, channels=1) as recorder:
            while not self.stop_event.is_set() and time.monotonic() < deadline:
                block = recorder.record(numframes=block_frames)
                mono = np.asarray(block, dtype=np.float32)
                if mono.ndim == 2:
                    mono = mono.mean(axis=1)
                rms = float(np.sqrt(np.mean(np.square(mono)))) if mono.size else 0.0
                threshold = gate.update(rms) if self.settings.silence_rms_threshold is None else self.settings.silence_rms_threshold
                levels.append(rms)
                if rms >= threshold:
                    voiced += 1

        if levels:
            array = np.asarray(levels, dtype=np.float64)
            self.report.blocks = len(levels)
            self.report.noise_floor = float(np.percentile(array, self.settings.noise_percentile))
            self.report.peak_block = float(array.max())
            self.report.voiced_ratio = voiced / len(levels)
        self.report.threshold = gate.threshold if self.settings.silence_rms_threshold is None else self.settings.silence_rms_threshold


class SourceRecorder(threading.Thread):
    """Records one source, writing an utterance per WAV file."""

    def __init__(
        self,
        source: SourceConfig,
        output_dir: Path,
        settings: SegmentSettings,
        stop_event: threading.Event,
        on_saved: Callable[[str, Path, float], None] | None = None,
        continuous: bool = False,
    ) -> None:
        super().__init__(name=f"record-{source.name}", daemon=True)
        self.source = source
        self.output_dir = output_dir
        self.settings = settings
        self.stop_event = stop_event
        self.on_saved = on_saved
        self.continuous = continuous
        self.continuous_path: Path | None = None
        self.continuous_seconds = 0.0
        self.peak_level = 0.0
        self.saved = 0
        self.error: Exception | None = None

    def run(self) -> None:
        try:
            self._record()
        except Exception as exc:  # surfaced by the CLI rather than dying silently
            self.error = exc

    def _record(self) -> None:
        segmenter = SilenceSegmenter(self.settings, self._save)
        recorder_source = resolve_recorder_source(self.source)
        block_frames = max(256, int(self.settings.sample_rate * 0.25))

        writer: ContinuousWriter | None = None
        if self.continuous:
            self.continuous_path = self.output_dir / "continuous" / continuous_filename(time.time(), self.source.name)
            writer = ContinuousWriter(self.continuous_path, self.settings.sample_rate)

        try:
            with recorder_source.recorder(samplerate=self.settings.sample_rate, channels=1) as recorder:
                while not self.stop_event.is_set():
                    block = recorder.record(numframes=block_frames)
                    mono = np.asarray(block, dtype=np.float32)
                    if mono.ndim == 2:
                        mono = mono.mean(axis=1)
                    if mono.size:
                        self.peak_level = max(self.peak_level, float(np.abs(mono).max()))
                    if writer is not None:
                        writer.write(mono)
                    segmenter.push(mono, time.time())
            segmenter.finish(time.time())
        finally:
            # Always close, otherwise the WAV keeps a header claiming zero
            # length and the whole session is unreadable.
            if writer is not None:
                writer.close()
                self.continuous_seconds = writer.seconds
                if writer.frames_written == 0:
                    self.continuous_path = None

    def _save(self, started_at: float, audio: np.ndarray) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / segment_filename(started_at, self.source.name)
        write_wav(path, audio, self.settings.sample_rate)
        self.saved += 1
        if self.on_saved is not None:
            self.on_saved(self.source.name, path, audio.size / self.settings.sample_rate)


class BrowserStreamRecorder:
    """Record float32 PCM pushed by the dashboard's tab-audio capture.

    Device sources own a thread that pulls samples from WASAPI. A browser tab
    has the opposite shape: Chrome owns capture and pushes samples into the
    local server. This class deliberately presents the small lifecycle surface
    used by ``RecordingManager`` (``start``, ``is_alive`` and ``join``), while
    feeding the same segmenter and continuous WAV writer as device capture.
    """

    def __init__(
        self,
        source: SourceConfig,
        output_dir: Path,
        settings: SegmentSettings,
        stop_event: threading.Event,
        on_saved: Callable[[str, Path, float], None] | None = None,
    ) -> None:
        self.source = source
        self.output_dir = output_dir
        self.settings = settings
        self.stop_event = stop_event
        self.on_saved = on_saved
        self.continuous = True
        self.continuous_path: Path | None = None
        self.continuous_seconds = 0.0
        self.peak_level = 0.0
        self.saved = 0
        self.error: Exception | None = None

        self._lock = threading.Lock()
        self._started = False
        self._finished = False
        self._sample_rate: int | None = None
        self._timeline_end: float | None = None
        self._segmenter: SilenceSegmenter | None = None
        self._writer: ContinuousWriter | None = None

    def start(self) -> None:
        with self._lock:
            self._started = True

    def is_alive(self) -> bool:
        with self._lock:
            return self._started and not self._finished and not self.stop_event.is_set()

    def join(self, timeout: float | None = None) -> None:  # noqa: ARG002 - thread-compatible API
        self.finish()

    def push_pcm(self, payload: bytes, sample_rate: int) -> int:
        """Accept little-endian float32 mono PCM and return its frame count."""
        if not 8_000 <= sample_rate <= 192_000:
            raise ValueError("Browser audio sample rate must be between 8000 and 192000 Hz")
        if not payload or len(payload) % 4:
            raise ValueError("Browser audio must contain complete float32 samples")
        if len(payload) > 8 * 1024 * 1024:
            raise ValueError("Browser audio chunk is too large")

        audio = np.frombuffer(payload, dtype="<f4").astype(np.float32, copy=True)
        if not np.all(np.isfinite(audio)):
            raise ValueError("Browser audio contains non-finite samples")

        with self._lock:
            if not self._started or self._finished or self.stop_event.is_set():
                raise RuntimeError("Chrome tab audio is not recording")
            if self._sample_rate is None:
                self._open(sample_rate, audio.size)
            elif sample_rate != self._sample_rate:
                raise ValueError(
                    f"Browser audio sample rate changed from {self._sample_rate} to {sample_rate} Hz"
                )

            if audio.size:
                self.peak_level = max(self.peak_level, float(np.abs(audio).max()))
            assert self._writer is not None
            assert self._segmenter is not None
            assert self._timeline_end is not None
            self._writer.write(audio)
            self._timeline_end += audio.size / sample_rate
            self._segmenter.push(audio, self._timeline_end)
            return int(audio.size)

    def finish(self) -> None:
        with self._lock:
            if self._finished:
                return
            self._finished = True
            now = self._timeline_end if self._timeline_end is not None else time.time()
            if self._segmenter is not None:
                self._segmenter.finish(now)
            if self._writer is not None:
                self._writer.close()
                self.continuous_seconds = self._writer.seconds
                if self._writer.frames_written == 0:
                    self.continuous_path = None

    def _open(self, sample_rate: int, first_frames: int) -> None:
        block_seconds = first_frames / sample_rate
        started_at = time.time() - block_seconds
        self._sample_rate = sample_rate
        self._timeline_end = started_at
        self.settings = replace(self.settings, sample_rate=sample_rate)
        self.continuous_path = self.output_dir / "continuous" / continuous_filename(started_at, self.source.name)
        self._writer = ContinuousWriter(self.continuous_path, sample_rate)
        self._segmenter = SilenceSegmenter(self.settings, self._save)

    def _save(self, started_at: float, audio: np.ndarray) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / segment_filename(started_at, self.source.name)
        write_wav(path, audio, self.settings.sample_rate)
        self.saved += 1
        if self.on_saved is not None:
            self.on_saved(self.source.name, path, audio.size / self.settings.sample_rate)
