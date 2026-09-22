"""Near-real-time transcription layered on the clip-per-utterance recorder.

This is a prototype whose only purpose is to measure, on a real machine, how
long after someone stops speaking their words appear. Latency is the thing that
decides whether a live view is worth building at all, and it cannot be guessed:
it depends on the model size, the device and how long the utterances are.

Nothing about recording changes. Every clip is still written to disk exactly as
before, with the same filename contract, so the ordinary offline pipeline
(better model, diarization, merging) can be run over the same session afterwards
and remains the source of truth. Live output is a read-only extra.

Transcription is decoupled from recording by a queue. The recorder thread must
never block or die - a stalled recorder loses audio that cannot be recovered,
and the WAV being written would be left unreadable - so `submit` only enqueues
and a separate worker thread does the slow work. If transcription cannot keep
up, the queue grows and lines arrive late; the recording is unaffected.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

from avilistener.file_transcriber import wav_to_audio_chunk

logger = logging.getLogger(__name__)

# Put on the queue to end the worker loop. A dedicated object is used rather
# than None so a genuine item can never be mistaken for the stop request.
_SENTINEL = object()


@dataclass(frozen=True)
class LiveLine:
    """One transcribed utterance, with the clock needed to judge latency."""

    source: str
    text: str
    started_at: float
    ended_at: float
    transcribed_at: float

    @property
    def latency(self) -> float:
        """Seconds between the speaker finishing and the text existing.

        Measured from the end of the clip, not from when it was queued: what a
        user perceives is the wait after they stop talking, which includes the
        silence the gate waits for before closing the clip.
        """
        return self.transcribed_at - self.ended_at


class LiveTranscriber:
    """Transcribes saved clips on a worker thread, as they are saved.

    `transcriber` is duck-typed: anything with `transcribe(chunk)` returning a
    result with `.text` (or None) works, which keeps the tests free of a real
    Whisper model.

    `on_done` is called with the clip's path once the worker has finished with
    it, whether it produced a line, was skipped as silent or failed. A caller
    that remembers which clips are already handled needs that moment and only
    that moment: a clip abandoned by `stop(drain=False)` is never announced, so
    it can be picked up again on the next session.
    """

    def __init__(
        self,
        transcriber,
        on_line: Callable[[LiveLine], None],
        on_error: Callable[[Path, Exception], None] | None = None,
        on_done: Callable[[Path], None] | None = None,
    ) -> None:
        self.transcriber = transcriber
        self.on_line = on_line
        self.on_error = on_error
        self.on_done = on_done

        self.lines_emitted = 0
        self.clips_skipped = 0
        self.latencies: list[float] = []

        self._queue: queue.Queue = queue.Queue()
        self._lock = threading.Lock()
        self._pending = 0
        self._abandon = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        # Daemon: a hung model load or transcription must never be the reason
        # the process refuses to exit after the user pressed Ctrl+C.
        self._thread = threading.Thread(target=self._run, name="live-transcribe", daemon=True)
        self._thread.start()

    def submit(self, source_name: str, path: Path, seconds: float) -> None:
        """Queue a clip. Signature matches `SourceRecorder.on_saved`.

        Called from the recorder thread, so it does nothing but enqueue.
        """
        del seconds  # duration is recomputed from the WAV when it is read
        with self._lock:
            self._pending += 1
        self._queue.put((source_name, Path(path)))

    def stop(self, drain: bool = True, timeout: float | None = None) -> None:
        """Stop the worker, optionally finishing what is already queued.

        Draining is the default because the last utterances of a session are
        usually the ones the user was waiting to see.
        """
        if self._thread is None:
            return
        if not drain:
            self._abandon.set()
        self._queue.put(_SENTINEL)
        self._thread.join(timeout=timeout)
        if not self._thread.is_alive():
            self._thread = None

    @property
    def pending(self) -> int:
        """Clips submitted but not yet finished, including the one in flight."""
        with self._lock:
            return self._pending

    @property
    def average_latency(self) -> float:
        with self._lock:
            return sum(self.latencies) / len(self.latencies) if self.latencies else 0.0

    @property
    def max_latency(self) -> float:
        with self._lock:
            return max(self.latencies) if self.latencies else 0.0

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is _SENTINEL:
                return
            if self._abandon.is_set():
                with self._lock:
                    self._pending -= 1
                continue
            source_name, path = item
            try:
                self._transcribe(source_name, path)
            except Exception as exc:  # one bad clip must not end the session
                if self.on_error is not None:
                    self.on_error(path, exc)
                else:
                    logger.warning("Live transcription failed for %s: %s", path, exc)
            finally:
                with self._lock:
                    self._pending -= 1
                if self.on_done is not None:
                    try:
                        self.on_done(path)
                    except Exception:  # bookkeeping must not end the worker
                        logger.warning("Live on_done failed for %s", path, exc_info=True)

    def _transcribe(self, source_name: str, path: Path) -> None:
        chunk = wav_to_audio_chunk(source_name, path)
        result = self.transcriber.transcribe(chunk)
        text = (getattr(result, "text", "") or "").strip() if result is not None else ""
        if not text:
            # Silence, or a clip the hallucination filters rejected. Counted so
            # a session that produces nothing can be told from one that was
            # never gated open in the first place.
            with self._lock:
                self.clips_skipped += 1
            return

        line = LiveLine(
            source=source_name,
            text=text,
            started_at=chunk.started_at,
            ended_at=chunk.ended_at,
            transcribed_at=time.time(),
        )
        with self._lock:
            self.lines_emitted += 1
            self.latencies.append(line.latency)
        self.on_line(line)


def format_live_line(line: LiveLine) -> str:
    """Render a line as `[HH:MM:SS] source (1.2s): text`.

    Local time, because the point is to read it while it happens. Kept pure so
    the format can be tested without a console.
    """
    stamp = datetime.fromtimestamp(line.started_at).strftime("%H:%M:%S")
    return f"[{stamp}] {line.source} ({line.latency:.1f}s): {line.text}"
