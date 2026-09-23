"""Live transcription for the dashboard: one meeting at a time, on demand.

Why this exists as its own thing rather than a hook inside the recorder.

**Live is independent of recording and of the offline pipeline.** The user
starts and stops it whenever they like, with whatever model they like, while a
recording is already running - or over clips that arrived from somewhere else
entirely. The offline transcript stays the source of truth; live output is a
read-only extra that can be thrown away and redone.

**Clips are discovered by scanning `recordings/`, not pushed by the recorder.**
A callback from `SourceRecorder` would only ever see clips this process wrote,
and the Discord receiver is a separate Node process writing into the same
directory - half a meeting would be missing. Discovery by filename timestamp
works for every producer, current and future, because the filename contract is
the one thing they all share. It also means live can be stopped, pointed at a
different model and resumed without touching a recorder thread, which must
never be disturbed: a stalled recorder loses audio that cannot be recovered.

**`live/` sits outside `recordings/`, and clips are copied into it.**
`recordings/` is strictly read-only for everything except the recorders: the
offline pipeline counts, hashes and compares those files to decide whether a
transcript is stale, so a single extra file in there would make every artifact
look out of date and could be transcribed a second time. Copying also decouples
the two lifetimes - the copy under `live/clips/` is what was actually fed to
the live model, whatever later happens to the original.

**A clip is eligible once it is old enough and its header is consistent.** Both
recorders write a whole clip in one go once the silence gate closes it (the
write itself was measured at 1-2 ms on this machine), so a file whose mtime is
at least a second old and whose declared frame count fits inside its own size
is finished. That pair of cheap checks is enough; a lock file or a rename
protocol would need every producer to cooperate, including the Node receiver.
Anything that raises while checking means "not ready" and is retried on the
next scan - never "seen", which would silently drop an utterance.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import threading
import time
import wave
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

from avilistener.file_transcriber import (
    sort_key_for_discord_wav,
    source_name_from_discord_wav,
    timestamp_from_discord_wav,
)
from avilistener.live import LiveLine, LiveTranscriber
from avilistener.server.workspace import Meeting

logger = logging.getLogger(__name__)

# Offered in the model picker. Small models exist here and not in the offline
# defaults because live trades accuracy for latency on purpose.
MODEL_OPTIONS = ["tiny", "base", "small", "medium", "large-v3", "large-v3-turbo"]

MODES = ("now", "catch_up")

SCAN_INTERVAL = 1.0
# A clip younger than this may still be being written. See the module docstring.
MIN_CLIP_AGE = 1.0
# How long a stop waits for the clip already inside the model. Bounded so the
# dashboard cannot hang on a request; a genuinely stuck worker is a daemon
# thread and dies with the process.
STOP_TIMEOUT = 60.0

_LINE_PATTERN = re.compile(r"^\[(\d{2}:\d{2}:\d{2})\] (.*?): (.*)$")


class LiveBusy(RuntimeError):
    """A live session is already running, here or for another meeting."""


class _AppendFailed(Exception):
    """A clip was transcribed but its line could not be written to the file."""


def meeting_key(meeting: Meeting) -> str:
    return f"{meeting.project.slug}/{meeting.slug}"


def model_options(default_model: str) -> list[str]:
    """The picker list, with the meeting's own model always selectable."""
    options = list(MODEL_OPTIONS)
    if default_model and default_model not in options:
        options.append(default_model)
    return options


def format_transcript_line(source: str, text: str, started_at: float) -> str:
    """`[HH:MM:SS] source: text`, local time of the clip's start.

    Local rather than UTC because the only reason to read this file is to
    follow a conversation that is happening now.
    """
    return f"[{clock_label(started_at)}] {source}: {text}"


def clock_label(started_at: float) -> str:
    """`HH:MM:SS` in local time, the one field a parsed-back line keeps.

    Named `clock_label` rather than `clock` because a session carries an
    injected `self.clock` time source, and one name for two unrelated things
    is how a test ends up asserting against the wrong one.
    """
    return datetime.fromtimestamp(started_at).strftime("%H:%M:%S")


def line_entry(
    index: int,
    time: str,
    source: str,
    text: str,
    started_at: float = 0.0,
    ended_at: float = 0.0,
    latency: float = 0.0,
) -> dict:
    """One line as the dashboard receives it.

    Built in one place because a line reconstructed from the transcript file
    and a line straight off the worker must have exactly the same keys: the
    interface renders both from the same list.
    """
    return {
        "index": index,
        "time": time,
        "source": source,
        "text": text,
        "started_at": started_at,
        "ended_at": ended_at,
        "latency": latency,
    }


def parse_transcript(path: Path) -> list[dict]:
    """Read back the lines a previous session appended.

    The clock a line was transcribed at is not in the file, so latencies come
    back as zero: they describe a session that is over and nobody is waiting on.
    """
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    lines: list[dict] = []
    for text in raw.splitlines():
        if not text.strip():
            continue
        match = _LINE_PATTERN.match(text)
        stamp, source, body = (match.group(1), match.group(2), match.group(3)) if match else ("", "", text)
        lines.append(line_entry(len(lines), stamp, source, body))
    return lines


def read_state(meeting: Meeting) -> dict:
    """The last session's watermark, model and handled clips."""
    try:
        payload = json.loads(meeting.live_state_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


@dataclass
class LiveStatus:
    """Exactly the JSON the dashboard polls for, in one place."""

    status: str = "stopped"  # stopped | loading | running | error
    model_size: str = ""
    default_model: str = ""
    model_options: list[str] = field(default_factory=list)
    mode: str | None = None
    watermark: float | None = None
    clips_seen: int = 0
    clips_transcribed: int = 0
    clips_skipped: int = 0
    pending: int = 0
    average_latency: float = 0.0
    max_latency: float = 0.0
    lines_total: int = 0
    lines: list[dict] = field(default_factory=list)
    error: str | None = None

    @property
    def active(self) -> bool:
        """Loading counts as active: the user pressed start and it is coming."""
        return self.status in ("loading", "running")

    def to_json(self, with_lines: bool = True) -> dict:
        return {
            "status": self.status,
            "active": self.active,
            "model_size": self.model_size,
            "default_model": self.default_model,
            "model_options": list(self.model_options),
            "mode": self.mode,
            "watermark": self.watermark,
            "clips_seen": self.clips_seen,
            "clips_transcribed": self.clips_transcribed,
            "clips_skipped": self.clips_skipped,
            "pending": self.pending,
            "average_latency": self.average_latency,
            "max_latency": self.max_latency,
            "lines_total": self.lines_total,
            # Always present, even when the caller asked for no lines: a list
            # that is sometimes missing is a trap for whoever renders it.
            "lines": list(self.lines) if with_lines else [],
            "error": self.error,
        }


class FakeLiveTranscriber:
    """A transcriber that loads no model. For e2e runs and dev only.

    Selected by `AVILISTENER_LIVE_FAKE_TRANSCRIBER=1` where the manager is
    built, so the flag is visible in one place and the offline pipeline can
    never pick it up. The sleep is there so the interface is exercised with a
    latency that is not zero.
    """

    def __init__(self, config: dict | None = None, delay: float = 0.2) -> None:
        self.config = dict(config or {})
        self.delay = delay

    def transcribe(self, chunk):
        from avilistener.transcriber import TranscriptResult

        time.sleep(self.delay)
        if chunk.audio is None or not len(chunk.audio):
            return None
        return TranscriptResult(
            source=chunk.source,
            text=f"simulated transcript of {chunk.source} clip",
            started_at=chunk.started_at,
            ended_at=chunk.ended_at,
            rms=chunk.rms,
        )


def build_transcriber(config: dict):
    """Default factory. Imported lazily: loading a model costs seconds."""
    from avilistener.transcriber import build_transcriber as _build

    return _build(config)


class LiveSession:
    """One live run over a meeting's clips: watermark, scanner and model.

    The model is loaded on this session's own thread, not in the request that
    started it: a cold large-v3 takes tens of seconds and the dashboard has to
    stay answerable, which is why the status goes `loading` -> `running`.
    """

    def __init__(
        self,
        meeting: Meeting,
        model_size: str,
        mode: str,
        config: dict,
        transcriber_factory: Callable[[dict], object],
        clock: Callable[[], float] = time.time,
        min_clip_age: float = MIN_CLIP_AGE,
        scan_interval: float = SCAN_INTERVAL,
        stop_timeout: float = STOP_TIMEOUT,
    ) -> None:
        if mode not in MODES:
            raise ValueError(f"Unknown live mode: {mode}")

        self.meeting = meeting
        self.key = meeting_key(meeting)
        self.model_size = model_size
        self.default_model = str(meeting.config.get("model_size") or "large-v3")
        self.mode = mode
        self.config = config
        self.transcriber_factory = transcriber_factory
        self.clock = clock
        self.min_clip_age = min_clip_age
        self.scan_interval = scan_interval
        self.stop_timeout = stop_timeout

        state = read_state(meeting)
        saved_watermark = state.get("watermark")
        if mode == "catch_up" and isinstance(saved_watermark, (int, float)):
            self.watermark = float(saved_watermark)
        else:
            self.watermark = float(clock())

        self._lock = threading.Lock()
        # A separate lock for `state.json`, held across build *and* write.
        # `self._lock` guards the scan hot path and every callback from the
        # worker, so holding it over disk I/O would stall discovery; a lock of
        # its own is what stops two writers (the worker's `_on_clip_done` and
        # the request thread's `stop`) from interleaving a stale payload over a
        # newer one. Order is always _state_lock -> _lock, never the reverse.
        self._state_lock = threading.Lock()
        self._seen: set[str] = {str(name) for name in state.get("seen") or []}
        # Filenames already complained about, so an unreadable name is logged
        # once rather than once per scan for the life of the session.
        self._unparsable: set[str] = set()
        # Submitted but not finished. Kept apart from `seen` so a clip dropped
        # by a stop is offered again rather than silently lost.
        self._queued: set[str] = set()
        # Transcribed but the line never reached the file. Not marked seen, so
        # the next scan offers the clip again instead of losing the line.
        self._unwritten: set[str] = set()
        # Previous sessions' lines are loaded so indexes stay stable across a
        # restart: the interface asks for everything after the last index it
        # saw, and that must not start over at zero.
        self._lines: list[dict] = parse_transcript(meeting.live_transcript_path)
        self._status = "loading"
        self._error: str | None = None
        self._final_counters = {
            "transcribed": 0,
            "skipped": 0,
            "pending": 0,
            "average_latency": 0.0,
            "max_latency": 0.0,
        }
        self.scans = 0  # visible so a caller can tell a scan really happened

        self._live: LiveTranscriber | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> None:
        if self._thread is not None:
            return
        self.meeting.live_clips_dir.mkdir(parents=True, exist_ok=True)
        self._persist_state()
        self._thread = threading.Thread(target=self._run, name="live-session", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop scanning, let the clip in the model finish, drop the queue."""
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=self.stop_timeout)

        # The reference is dropped only after the drain: `snapshot` reads it,
        # and nulling it first made the dashboard show zeros for as long as the
        # clip already inside the model took to finish - up to a minute.
        with self._lock:
            live = self._live
        if live is not None:
            # drain=False: queued clips are not marked seen, so `catch_up`
            # picks them up instead of the user waiting out a backlog.
            live.stop(drain=False, timeout=self.stop_timeout)
            self._final_counters = {
                "transcribed": live.lines_emitted,
                "skipped": live.clips_skipped,
                "pending": 0,
                "average_latency": live.average_latency,
                "max_latency": live.max_latency,
            }
            self._live = None
        with self._lock:
            if self._status != "error":
                self._status = "stopped"
        self._persist_state()

    @property
    def active(self) -> bool:
        with self._lock:
            return self._status in ("loading", "running")

    @property
    def status_name(self) -> str:
        with self._lock:
            return self._status

    # -- the scan thread ---------------------------------------------------
    def _run(self) -> None:
        try:
            transcriber = self.transcriber_factory(self.config)
        except Exception as exc:  # a missing model or a dead GPU ends here
            logger.warning("Live model load failed for %s: %s", self.key, exc)
            with self._lock:
                self._status = "error"
                self._error = str(exc) or exc.__class__.__name__
            return

        live = LiveTranscriber(
            transcriber,
            on_line=self._on_line,
            on_error=self._on_clip_error,
            on_done=self._on_clip_done,
        )
        # A stop that gave up waiting for a slow load (a first-use download can
        # take minutes) has already reported `stopped`. Publishing a worker now
        # would flip the session back to `running` with nothing left to stop
        # it, holding the model and the slot. Checked under the lock `stop`
        # reads `_live` under, so one of the two always sees the other.
        with self._lock:
            if self._stop.is_set():
                return
            live.start()
            self._live = live
            self._status = "running"

        while not self._stop.is_set():
            try:
                self._scan()
            except Exception:  # a scan must never be the reason live dies
                logger.warning("Live scan failed for %s", self.key, exc_info=True)
            # Waiting on the event rather than sleeping is what makes a stop
            # prompt instead of up to a full scan period late.
            self._stop.wait(self.scan_interval)

    def _scan(self) -> None:
        # A stop that timed out drops the transcriber while this loop may still
        # be between iterations, so the reference is taken once and checked.
        live = self._live
        directory = self.meeting.recordings_dir
        if live is None or not directory.exists():
            self.scans += 1
            return
        # Top level only: `continuous/` holds hour-long session files and
        # `processed/` holds legacy clips an older offline pipeline moved
        # there. Nothing moves clips out of the top level any more - the
        # dashboard pipeline only reads them - so every new clip is here.
        # Sorted by the timestamp in the name, not by the name itself: the
        # Discord receiver and the local recorder use different prefixes, so
        # plain name order would transcribe an interleaved meeting per source.
        for path in sorted(directory.glob("*.wav"), key=_scan_order):
            if self._stop.is_set():
                break
            if not self._eligible(path):
                continue
            copy = self._copy_for_live(path)
            if copy is None:
                continue
            with self._lock:
                self._queued.add(path.name)
            live.submit(source_name_from_discord_wav(path), copy, 0.0)
        self.scans += 1

    def _eligible(self, path: Path) -> bool:
        name = path.name
        if name.endswith("-continuous.wav"):
            return False
        with self._lock:
            if name in self._seen or name in self._queued:
                return False
        started_at = timestamp_from_discord_wav(path)
        if started_at is None:
            self._warn_unparsable(name)
            return False
        if started_at < self.watermark:
            return False
        try:
            if not path.is_file():
                return False
            stat = path.stat()
            if self.clock() - stat.st_mtime < self.min_clip_age:
                return False
            return _wav_is_complete(path, stat.st_size)
        except Exception:
            # Being written, locked by the other process, half a header, or
            # zero bytes (which `wave.open` reports as EOFError, not wave.Error).
            # A clip that cannot be inspected is by definition not ready, and
            # anything narrower would abort the rest of the scan cycle over one
            # bad file. Not marked seen: the next scan tries again.
            return False

    def _warn_unparsable(self, name: str) -> None:
        """Say so once per filename: a clip nobody can date is never picked up.

        Silence here is the worst outcome - the utterance is simply missing
        from the live transcript with nothing anywhere to explain the gap.
        """
        with self._lock:
            if name in self._unparsable:
                return
            self._unparsable.add(name)
        logger.warning(
            "Live: ignoring %s in %s - no timestamp in the filename, so it cannot be placed "
            "against the watermark",
            name,
            self.key,
        )

    def _copy_for_live(self, path: Path) -> Path | None:
        destination = self.meeting.live_clips_dir / path.name
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists() and destination.stat().st_size == path.stat().st_size:
                return destination
            shutil.copy2(path, destination)
        except OSError as exc:
            logger.warning("Could not copy %s for live transcription: %s", path, exc)
            return None
        return destination

    # -- callbacks from the transcriber worker -----------------------------
    def _on_line(self, line: LiveLine) -> None:
        # File first: the file is what outlives the session, so a line the
        # dashboard shows must already be in it. Raising here reaches
        # `_on_clip_error` with the clip's path, which is what keeps the clip
        # out of `seen`.
        try:
            self._append_transcript(line)
        except OSError as exc:
            raise _AppendFailed(str(exc)) from exc
        with self._lock:
            self._lines.append(
                line_entry(
                    index=len(self._lines),
                    time=clock_label(line.started_at),
                    source=line.source,
                    text=line.text,
                    started_at=line.started_at,
                    ended_at=line.ended_at,
                    latency=line.latency,
                )
            )

    def _on_clip_error(self, path: Path, exc: Exception) -> None:
        if isinstance(exc, _AppendFailed):
            logger.warning("Could not append the line for %s, will retry: %s", path, exc)
            with self._lock:
                self._unwritten.add(path.name)
            return
        logger.warning("Live transcription failed for %s: %s", path, exc)

    def _on_clip_done(self, path: Path) -> None:
        """The clip left the worker: remember it and write that down.

        Done means done in every sense - a line, a silent skip or a failed
        transcription - because all three are settled. Two are not: a clip
        abandoned by a stop, which is never announced here, and one whose
        line could not be written, which goes back to the scan for a retry.
        """
        with self._lock:
            self._queued.discard(path.name)
            if path.name in self._unwritten:
                self._unwritten.discard(path.name)
                return
            self._seen.add(path.name)
        self._persist_state()

    # -- persistence -------------------------------------------------------
    def _append_transcript(self, line: LiveLine) -> None:
        """One open/append/close per line.

        Same reason as the job logs: an external agent tails this file while
        the meeting runs, and a buffered write would show it nothing until the
        session ended. Raises `OSError`; the caller decides what a lost line means.
        """
        path = self.meeting.live_transcript_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(format_transcript_line(line.source, line.text, line.started_at) + "\n")

    def _persist_state(self) -> None:
        """Replace `state.json` atomically, one writer at a time.

        Two threads write it - the worker as each clip finishes and the request
        thread on stop - and a reader can arrive at any moment. A plain
        `write_text` truncates first, so a reader could see an empty or partial
        file, and two interleaved writers could leave the older `seen` set on
        disk. Writing a sibling temp file and `os.replace`-ing it makes the
        swap atomic for readers, and `_state_lock` (held across build and
        write, so the payload cannot be overtaken) makes it atomic for writers.
        """
        path = self.meeting.live_state_path
        temp = path.with_name(path.name + ".tmp")
        with self._state_lock:
            with self._lock:
                payload = {
                    "watermark": self.watermark,
                    "model_size": self.model_size,
                    "seen": sorted(self._seen),
                    "lines_total": len(self._lines),
                }
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
                _replace_when_free(temp, path)
            except OSError as exc:
                logger.warning("Could not write %s: %s", path, exc)
                try:
                    temp.unlink(missing_ok=True)
                except OSError:
                    pass

    # -- reporting ---------------------------------------------------------
    def snapshot(self, after: int = 0) -> LiveStatus:
        live = self._live
        counters = dict(self._final_counters)
        if live is not None:
            counters = {
                "transcribed": live.lines_emitted,
                "skipped": live.clips_skipped,
                "pending": live.pending,
                "average_latency": live.average_latency,
                "max_latency": live.max_latency,
            }
        with self._lock:
            lines = [line for line in self._lines if line["index"] >= after]
            return LiveStatus(
                status=self._status,
                model_size=self.model_size,
                default_model=self.default_model,
                model_options=model_options(self.default_model),
                mode=self.mode,
                watermark=self.watermark,
                clips_seen=len(self._seen),
                clips_transcribed=counters["transcribed"],
                clips_skipped=counters["skipped"],
                pending=counters["pending"],
                average_latency=counters["average_latency"],
                max_latency=counters["max_latency"],
                lines_total=len(self._lines),
                lines=lines,
                error=self._error,
            )


def _replace_when_free(temp: Path, path: Path, attempts: int = 40, pause: float = 0.005) -> None:
    """`os.replace`, retried while a reader still holds the destination open.

    On Windows the swap fails with a sharing violation for as long as anything
    has the target open, and the dashboard polls `state.json` - so the very act
    of watching a live session could throw a write away. A lost write means a
    clip is transcribed again after a restart, which is exactly what `seen`
    exists to prevent, so a fifth of a second of patience is worth the retry.
    On POSIX the first attempt always succeeds and this costs nothing.
    """
    for attempt in range(attempts):
        try:
            os.replace(temp, path)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(pause)


def _scan_order(path: Path) -> tuple[float, str]:
    """`sort_key_for_discord_wav`, tolerant of a clip that vanished mid-scan.

    The key falls back to `st_mtime` for a name it cannot parse, and a file
    deleted or moved between the glob and the sort would raise there - losing
    the whole cycle over a clip that is no longer any of our business.
    """
    try:
        return sort_key_for_discord_wav(path)
    except OSError:
        return (0.0, path.name)


def _wav_is_complete(path: Path, size: int) -> bool:
    """Whether the file holds the audio its own header claims.

    A clip caught mid-write has a header describing more frames than are there;
    reading it would either fail or transcribe silence as if it were the end of
    the utterance.
    """
    with wave.open(str(path), "rb") as handle:
        frames = handle.getnframes()
        channels = handle.getnchannels()
        width = handle.getsampwidth()
    if frames <= 0 or channels <= 0 or width <= 0:
        return False
    return size >= 44 + frames * channels * width


class LiveManager:
    """One live session at a time, for the whole dashboard.

    Global rather than per meeting because the thing being shared is the GPU:
    two models loaded at once is how both sessions end up slower than either
    would have been alone, and the user only watches one meeting anyway.
    """

    def __init__(
        self,
        transcriber_factory: Callable[[dict], object] = build_transcriber,
        clock: Callable[[], float] = time.time,
        min_clip_age: float = MIN_CLIP_AGE,
        scan_interval: float = SCAN_INTERVAL,
        stop_timeout: float = STOP_TIMEOUT,
    ) -> None:
        self.transcriber_factory = transcriber_factory
        self.clock = clock
        self.min_clip_age = min_clip_age
        self.scan_interval = scan_interval
        self.stop_timeout = stop_timeout
        # The last session is kept after it ends so its model, watermark and
        # error message still have somewhere to be read from.
        self._session: LiveSession | None = None
        self._lock = threading.Lock()

    @property
    def session(self) -> LiveSession | None:
        return self._session

    def start(self, meeting: Meeting, model_size: str | None = None, mode: str = "now") -> dict:
        mode = mode or "now"
        if mode not in MODES:
            raise ValueError(f"Unknown live mode: {mode}")

        with self._lock:
            active = self._session is not None and self._session.active
            if active:
                raise LiveBusy(f"Live transcription is already running for {self._session.key}")

            from avilistener.processing import transcription_config

            config = transcription_config(meeting)
            chosen = str(model_size or config.get("model_size") or "large-v3")
            config["model_size"] = chosen

            session = LiveSession(
                meeting=meeting,
                model_size=chosen,
                mode=mode,
                config=config,
                transcriber_factory=self.transcriber_factory,
                clock=self.clock,
                min_clip_age=self.min_clip_age,
                scan_interval=self.scan_interval,
                stop_timeout=self.stop_timeout,
            )
            session.start()
            self._session = session
        return self.status(meeting, with_lines=False)

    def stop(self, meeting: Meeting) -> dict:
        with self._lock:
            session = self._session
            if session is not None and session.key == meeting_key(meeting) and session.active:
                session.stop()
        return self.status(meeting, with_lines=False)

    def status(self, meeting: Meeting, after: int = 0, with_lines: bool = True) -> dict:
        session = self._session
        if session is not None and session.key == meeting_key(meeting):
            return session.snapshot(after).to_json(with_lines=with_lines)
        return self._status_from_disk(meeting, after).to_json(with_lines=with_lines)

    def global_status(self) -> dict:
        session = self._session
        active = session is not None and session.active
        return {
            "active": active,
            "meeting": session.key if (session is not None and active) else None,
            "status": session.status_name if session is not None else "stopped",
        }

    def transcript_text(self, meeting: Meeting) -> str:
        try:
            return meeting.live_transcript_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""

    def _status_from_disk(self, meeting: Meeting, after: int = 0) -> LiveStatus:
        """What the dashboard shows after a server restart.

        Nothing is running, but the transcript file and `state.json` still
        describe the last session, so the lines do not vanish from the screen
        just because the process did.
        """
        state = read_state(meeting)
        default_model = str(meeting.config.get("model_size") or "large-v3")
        watermark = state.get("watermark")
        lines = parse_transcript(meeting.live_transcript_path)
        return LiveStatus(
            status="stopped",
            model_size=str(state.get("model_size") or default_model),
            default_model=default_model,
            model_options=model_options(default_model),
            mode=None,
            watermark=float(watermark) if isinstance(watermark, (int, float)) else None,
            clips_seen=len(state.get("seen") or []),
            lines_total=len(lines),
            lines=[line for line in lines if line["index"] >= after],
        )
