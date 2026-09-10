"""Background work for the dashboard: recording sessions and one-off jobs.

Recording runs in this process rather than as a child process on purpose.
Stopping has to be graceful: the continuous WAV's header is only finalised on
close, so a killed process leaves a file claiming zero length and the whole
session becomes unreadable. Windows has no way to deliver a polite signal to a
child, so the recorder threads are owned here and stopped with an event.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from avilistener.audio import SourceConfig
from avilistener.recorder import BrowserStreamRecorder, SegmentSettings, SourceRecorder


def append_log(path: Path | None, line: str) -> None:
    """Append one line, reopening each time.

    Logs are written as they happen rather than buffered and flushed at the
    end: the reason to read them is usually that something crashed, and a
    buffered log loses exactly the part that explains why.
    """
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with path.open("a", encoding="utf-8") as handle:
            handle.write(f"[{stamp}] {line.rstrip()}\n")
    except OSError:
        pass  # logging must never take down the thing it is logging


@dataclass
class Job:
    id: str
    kind: str
    meeting: str
    status: str = "running"  # running | done | error
    log: list[str] = field(default_factory=list)
    error: str | None = None
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    log_path: Path | None = None

    def to_json(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "meeting": self.meeting,
            "status": self.status,
            "log": self.log[-200:],
            "error": self.error,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "log_file": self.log_path.name if self.log_path else None,
        }


class JobRegistry:
    """Runs one long task at a time per meeting and keeps its log."""

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def active_for(self, meeting: str) -> Job | None:
        with self._lock:
            for job in self._jobs.values():
                if job.meeting == meeting and job.status == "running":
                    return job
        return None

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def recent(self, meeting: str | None = None, limit: int = 20) -> list[Job]:
        with self._lock:
            jobs = list(self._jobs.values())
        if meeting is not None:
            jobs = [job for job in jobs if job.meeting == meeting]
        return sorted(jobs, key=lambda job: job.started_at, reverse=True)[:limit]

    def start(
        self,
        kind: str,
        meeting: str,
        work: Callable[[Callable[[str], None]], None],
        log_dir: Path | None = None,
    ) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], kind=kind, meeting=meeting)
        if log_dir is not None:
            job.log_path = log_dir / f"{time.strftime('%Y-%m-%dT%H-%M-%S')}-{kind}.log"
        with self._lock:
            # Check and reserve together: two tabs may submit at the same time.
            existing = next((item for item in self._jobs.values() if item.meeting == meeting and item.status == "running"), None)
            if existing is not None:
                raise RuntimeError(f"{existing.kind} is already running for this meeting")
            self._jobs[job.id] = job

        def append(line: str) -> None:
            job.log.append(line.rstrip())
            append_log(job.log_path, line)

        def run() -> None:
            append(f"{kind} started")
            try:
                work(append)
                job.status = "done"
                append(f"{kind} finished")
            # BaseException, not Exception: CLI-flavoured code inside a job
            # signals failure with SystemExit, which `except Exception` lets
            # through - the thread dies silently and the job shows "running"
            # forever with no error.
            except BaseException as exc:
                job.status = "error"
                job.error = str(exc) or exc.__class__.__name__
                # The last frame names the actual failure; the full traceback
                # goes to the file so the console stays readable.
                job.log.append(traceback.format_exc().strip().splitlines()[-1])
                append_log(job.log_path, f"{kind} FAILED: {job.error}")
                append_log(job.log_path, traceback.format_exc())
            finally:
                job.finished_at = time.time()

        threading.Thread(target=run, name=f"job-{kind}", daemon=True).start()
        return job


@dataclass
class RecordingSession:
    STALL_AFTER_SECONDS = 4.0

    meeting: str
    output_dir: Path
    sources: list[str]
    started_at: float
    stop_event: threading.Event
    recorders: list[SourceRecorder | BrowserStreamRecorder]
    _observed_sizes: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _last_growth_at: dict[str, float] = field(default_factory=dict, init=False, repr=False)
    _monitor_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    @property
    def running(self) -> bool:
        return any(recorder.is_alive() for recorder in self.recorders)

    def status(self) -> dict:
        now = time.time()
        return {
            "meeting": self.meeting,
            "running": self.running,
            "sources": self.sources,
            "started_at": self.started_at,
            "elapsed": now - self.started_at,
            "segments": {recorder.source.name: recorder.saved for recorder in self.recorders},
            "peaks": {recorder.source.name: round(recorder.peak_level, 4) for recorder in self.recorders},
            "continuous_files": self._continuous_file_status(now),
            "errors": {
                recorder.source.name: str(recorder.error)
                for recorder in self.recorders
                if recorder.error is not None
            },
        }

    def _continuous_file_status(self, now: float) -> dict[str, dict]:
        """Live on-disk growth for every source's full-session WAV.

        Silence is still written to continuous files, so a healthy source must
        keep growing even when nobody speaks. That makes file growth a stronger
        recording-health signal than peak level or utterance count.
        """
        result: dict[str, dict] = {}
        session_running = self.running
        with self._monitor_lock:
            for recorder in self.recorders:
                name = recorder.source.name
                path = recorder.continuous_path
                size = _file_size(path)
                previous = self._observed_sizes.get(name)

                if previous is None:
                    self._last_growth_at[name] = now if size else self.started_at
                elif size > previous:
                    self._last_growth_at[name] = now
                self._observed_sizes[name] = size

                if recorder.error is not None:
                    state = "failed"
                elif not session_running:
                    state = "complete" if size else "empty"
                elif size == 0 and now - self.started_at < self.STALL_AFTER_SECONDS:
                    state = "waiting"
                elif now - self._last_growth_at[name] >= self.STALL_AFTER_SECONDS:
                    state = "stalled"
                else:
                    state = "growing"

                result[name] = {
                    "filename": path.name if path is not None and path.exists() else None,
                    "bytes": size,
                    "state": state,
                }
        return result


def _file_size(path: Path | None) -> int:
    if path is None:
        return 0
    try:
        return path.stat().st_size
    except OSError:
        return 0


def discord_token_in_environment() -> bool:
    """Whether DISCORD_BOT_TOKEN is already available to this process.

    The receiver has always read the token from the environment, so a machine
    set up for the command line needs nothing pasted into the interface.
    """
    return bool(os.environ.get("DISCORD_BOT_TOKEN", "").strip())


def list_voice_channels(token: str) -> list[dict]:
    """Voice channels the bot can see, grouped by server.

    Asked of Discord directly rather than making the user hunt for a channel id
    with Developer Mode. Uses urllib so the dashboard gains no dependency for
    two occasional requests.
    """
    import urllib.error
    import urllib.request

    bearer = (token or os.environ.get("DISCORD_BOT_TOKEN", "")).strip()
    if not bearer:
        raise RuntimeError("No Discord bot token available")

    def get(path: str):
        request = urllib.request.Request(
            f"https://discord.com/api/v10{path}",
            headers={"Authorization": f"Bot {bearer}", "User-Agent": "AviListener (local, 0.1)"},
        )
        with urllib.request.urlopen(request, timeout=15) as response:
            return json.loads(response.read().decode("utf-8"))

    try:
        guilds = get("/users/@me/guilds")
    except urllib.error.HTTPError as exc:
        if exc.code == 401:
            raise RuntimeError("Discord rejected the bot token")
        raise RuntimeError(f"Discord returned {exc.code} listing servers")
    except OSError as exc:
        raise RuntimeError(f"Could not reach Discord: {exc}")

    servers = []
    for guild in guilds:
        try:
            channels = get(f"/guilds/{guild['id']}/channels")
        except (urllib.error.HTTPError, OSError):
            continue  # bot may lack permission on one server; skip it
        # Type 2 is a voice channel; 13 is a stage channel, which cannot be
        # joined the same way and is left out.
        voice = [
            {"id": str(channel["id"]), "name": str(channel.get("name") or channel["id"])}
            for channel in channels
            if channel.get("type") == 2
        ]
        if voice:
            servers.append({"guild": str(guild.get("name") or guild["id"]), "channels": voice})
    return servers


class DiscordReceiver:
    """Runs the Node receiver, which writes one WAV per person who speaks.

    Discord identifies who is talking, so its files are already per-participant
    and need no diarization. The filenames match what the transcriber parses,
    so each Discord username becomes the speaker label directly.
    """

    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root
        self.process: subprocess.Popen | None = None
        self.log: list[str] = []
        self._reader: threading.Thread | None = None
        # Which meeting the receiver is (or last was) recording for, so the
        # interface can show it as recording without any local audio source.
        self.meeting: str | None = None
        self.started_at: float | None = None

    @property
    def running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def start(self, output_dir: Path, token: str, channel_id: str | None, meeting: str | None = None) -> None:
        script = self.project_root / "discord-receiver" / "src" / "index.js"
        if not script.exists():
            raise RuntimeError("discord-receiver is missing from this checkout")
        if not token.strip() and not discord_token_in_environment():
            raise RuntimeError(
                "No Discord bot token. Set DISCORD_BOT_TOKEN in your environment, or paste one in settings."
            )

        config_path = output_dir.parent / "discord-receiver.yaml"
        lines = [
            "token_env: DISCORD_BOT_TOKEN",
            'command_prefix: "!"',
            f"output_dir: {json.dumps(str(output_dir))}",
            "silence_rms_threshold: 0.006",
            "silence_duration_ms: 1500",
            "preroll_ms: 300",
            "min_segment_ms: 700",
        ]
        if channel_id and str(channel_id).strip():
            lines.append(f"auto_join_voice_channel_id: {json.dumps(str(channel_id).strip())}")
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        env = dict(os.environ)
        # A token saved in settings wins; otherwise the child simply inherits
        # DISCORD_BOT_TOKEN, which is how this was already being run by hand.
        if token.strip():
            env["DISCORD_BOT_TOKEN"] = token.strip()
        self.log = []
        self.meeting = meeting
        self.started_at = time.time()
        self.process = subprocess.Popen(
            ["node", str(script), str(config_path)],
            cwd=str(self.project_root),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        def pump() -> None:
            assert self.process and self.process.stdout
            for line in self.process.stdout:
                # The token is never echoed by the receiver, but keep the log
                # bounded so a long call cannot grow memory without limit.
                self.log.append(line.rstrip())
                del self.log[:-200]

        self._reader = threading.Thread(target=pump, name="discord-log", daemon=True)
        self._reader.start()

    def stop(self) -> None:
        if self.process is None:
            return
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                self.process.kill()
        self.process = None

    def status(self) -> dict:
        running = self.running
        return {
            "running": running,
            "meeting": self.meeting,
            "started_at": self.started_at,
            "elapsed": (time.time() - self.started_at) if running and self.started_at else None,
            "log": self.log[-40:],
        }


class RecordingManager:
    """One recording at a time, owned in-process so it can stop cleanly."""

    SOURCE_KINDS = {"mic": "microphone", "system": "loopback", "chrome": "browser", "discord": "loopback"}

    def __init__(self) -> None:
        self._session: RecordingSession | None = None
        self._log_path: Path | None = None
        self._lock = threading.Lock()

    @property
    def session(self) -> RecordingSession | None:
        return self._session

    def status(self) -> dict | None:
        session = self._session
        return session.status() if session is not None else None

    def start(
        self,
        meeting_key: str,
        output_dir: Path,
        sources: list[str],
        sample_rate: int,
        devices: dict[str, str | None],
        log_dir: Path | None = None,
    ) -> dict:
        with self._lock:
            if self._session is not None and self._session.running:
                raise RuntimeError("A recording is already running")
            if not sources:
                raise ValueError("Pick at least one source to record")
            if {"system", "chrome"}.issubset(sources):
                raise ValueError("Pick either PC audio or Chrome tab, not both")

            output_dir.mkdir(parents=True, exist_ok=True)
            stop_event = threading.Event()
            settings = SegmentSettings(sample_rate=sample_rate)
            recorders = []
            for name in sources:
                kind = self.SOURCE_KINDS.get(name)
                if kind is None:
                    raise ValueError(f"Unknown source: {name}")
                source = SourceConfig(name=name, kind=kind, enabled=True, device=devices.get(name))
                if kind == "browser":
                    recorders.append(BrowserStreamRecorder(source, output_dir, settings, stop_event))
                else:
                    recorders.append(SourceRecorder(source, output_dir, settings, stop_event, continuous=True))

            for recorder in recorders:
                recorder.start()

            self._log_path = (log_dir / f"{time.strftime('%Y-%m-%dT%H-%M-%S')}-recording.log") if log_dir else None
            append_log(self._log_path, f"recording started: {', '.join(sources)}")
            for name in sources:
                target = "tab selected in Chrome" if name == "chrome" else devices.get(name) or "system default"
                append_log(self._log_path, f"  {name} -> {target}")

            self._session = RecordingSession(
                meeting=meeting_key,
                output_dir=output_dir,
                sources=list(sources),
                started_at=time.time(),
                stop_event=stop_event,
                recorders=recorders,
            )
            return self._session.status()

    def push_browser_audio(self, meeting_key: str, payload: bytes, sample_rate: int) -> int:
        """Feed one browser PCM chunk into the active meeting recording."""
        with self._lock:
            session = self._session
            if session is None or session.meeting != meeting_key or not session.running:
                raise RuntimeError("This meeting is not recording")
            browser = next(
                (item for item in session.recorders if isinstance(item, BrowserStreamRecorder)),
                None,
            )
        if browser is None:
            raise RuntimeError("Chrome tab is not selected as a recording source")
        return browser.push_pcm(payload, sample_rate)

    def stop(self) -> dict:
        with self._lock:
            session = self._session
            if session is None:
                raise RuntimeError("Nothing is recording")

            session.stop_event.set()
            for recorder in session.recorders:
                # Joining matters: the continuous WAV header is rewritten on
                # close, and reporting "stopped" before that finishes would
                # invite reading a file that is still being finalised.
                recorder.join(timeout=10)

            status = session.status()
            status["running"] = False

            for recorder in session.recorders:
                name = recorder.source.name
                append_log(
                    self._log_path,
                    f"{name}: {recorder.saved} clip(s), {recorder.continuous_seconds:.0f}s continuous, "
                    f"peak {recorder.peak_level:.4f}",
                )
                if recorder.error is not None:
                    append_log(self._log_path, f"{name} FAILED: {recorder.error}")
                elif recorder.peak_level < 0.01:
                    # Near-silence almost always means the wrong device, and it
                    # is invisible until a transcript comes back empty.
                    append_log(self._log_path, f"{name} WARNING: captured almost no audio, check the device")
            append_log(self._log_path, "recording stopped")

            self._session = None
            return status
