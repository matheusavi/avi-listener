"""Live transcription as the dashboard drives it: discovery, state and routes.

Everything here runs against a fake transcriber with an injected clock and age
threshold, so no model is ever loaded and no test waits out a real second. What
is being checked is the part that can silently go wrong: which clips are picked
up, that none is transcribed twice, that a half-written clip is retried rather
than lost, and that `recordings/` comes out byte for byte as it went in.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

import numpy as np
import pytest
from fastapi import HTTPException

from avilistener.file_transcriber import collect_wav_clips
from avilistener.recorder import continuous_filename, segment_filename, write_wav
from avilistener.server import app as app_module
from avilistener.server.live import (
    MODEL_OPTIONS,
    FakeLiveTranscriber,
    LiveBusy,
    LiveManager,
    clock,
    parse_transcript,
)
from avilistener.server.workspace import Workspace
from avilistener.timeline import sha256
from avilistener.transcriber import TranscriptResult

BASE = 1_700_000_000.0
TIMEOUT = 10.0

CONTRACT_KEYS = {
    "status",
    "active",
    "model_size",
    "default_model",
    "model_options",
    "mode",
    "watermark",
    "clips_seen",
    "clips_transcribed",
    "clips_skipped",
    "pending",
    "average_latency",
    "max_latency",
    "lines_total",
    "lines",
    "error",
}


class Clock:
    """A clock the test moves by hand, so nothing has to sleep a real second."""

    def __init__(self, now: float = BASE) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class CountingTranscriber:
    """Records every clip it is handed, so double work is visible."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.clips: list[str] = []

    def transcribe(self, chunk) -> TranscriptResult | None:
        with self.lock:
            self.clips.append(chunk.source)
        if not chunk.audio.size:
            return None
        return TranscriptResult(
            source=chunk.source,
            text=f"text from {chunk.source}",
            started_at=chunk.started_at,
            ended_at=chunk.ended_at,
            rms=chunk.rms,
        )

    @property
    def count(self) -> int:
        with self.lock:
            return len(self.clips)


def tone(seconds: float, amplitude: float = 0.4, sample_rate: int = 16000) -> np.ndarray:
    frames = int(sample_rate * seconds)
    t = np.arange(frames) / sample_rate
    return (np.sin(2 * np.pi * 440 * t) * amplitude).astype(np.float32)


def write_clip(directory: Path, source: str, started_at: float, seconds: float = 0.3) -> Path:
    """Write a clip exactly as a recorder would, with a matching mtime."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / segment_filename(started_at, source)
    write_wav(path, tone(seconds), 16000)
    os.utime(path, (started_at, started_at))
    return path


def truncate(path: Path, drop: int = 5000) -> None:
    """Leave the header claiming more frames than the file now holds."""
    data = path.read_bytes()
    mtime = path.stat().st_mtime
    path.write_bytes(data[:-drop])
    os.utime(path, (mtime, mtime))


def wait_for(predicate, message: str, timeout: float = TIMEOUT) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError(message)


def fingerprint(directory: Path) -> dict[str, tuple[int, str]]:
    """Every file under a directory, with its size and content hash."""
    return {
        str(path.relative_to(directory)): (path.stat().st_size, sha256(path))
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


@pytest.fixture()
def env(tmp_path: Path):
    """A workspace, a meeting, a hand-wound clock and a live manager."""

    class Env:
        def __init__(self) -> None:
            self.workspace = Workspace(root=tmp_path / "workspace")
            self.project = self.workspace.create_project("Work")
            self.meeting = self.project.create_meeting("Standup")
            self.other = self.project.create_meeting("Retro")
            self.clock = Clock()
            self.transcriber = CountingTranscriber()
            self.managers: list[LiveManager] = []

        def manager(self, factory=None) -> LiveManager:
            manager = LiveManager(
                transcriber_factory=factory or (lambda config: self.transcriber),
                clock=self.clock,
                min_clip_age=1.0,
                scan_interval=0.02,
                stop_timeout=TIMEOUT,
            )
            self.managers.append(manager)
            return manager

        def clip(self, source: str, started_at: float, seconds: float = 0.3, meeting=None) -> Path:
            target = (meeting or self.meeting).recordings_dir
            return write_clip(target, source, started_at, seconds)

        def wait_running(self, manager: LiveManager, meeting=None) -> None:
            meeting = meeting or self.meeting
            wait_for(
                lambda: manager.status(meeting)["status"] == "running",
                "the session never reached running",
            )

        def wait_scans(self, manager: LiveManager, count: int = 3) -> None:
            """Wait for whole scan passes, so 'was never picked up' is real."""
            session = manager.session
            assert session is not None
            target = session.scans + count
            wait_for(lambda: session.scans >= target, "the scan loop stopped running")

    env = Env()
    yield env
    for manager in env.managers:
        session = manager.session
        if session is not None and session.active:
            session.stop()


def state_of(meeting) -> dict:
    return json.loads(meeting.live_state_path.read_text(encoding="utf-8"))


class TestDiscovery:
    def test_only_clips_at_or_after_the_watermark_are_transcribed(self, env) -> None:
        old = env.clip("mic", BASE - 60)
        fresh = env.clip("mic", BASE + 10)

        manager = env.manager()
        manager.start(env.meeting)  # mode="now": watermark is the clock, BASE
        env.wait_running(manager)
        env.clock.advance(30)

        wait_for(
            lambda: manager.status(env.meeting)["lines_total"] == 1,
            "the clip after the watermark was never transcribed",
        )
        env.wait_scans(manager)

        copied = sorted(path.name for path in env.meeting.live_clips_dir.glob("*.wav"))
        assert copied == [fresh.name]
        assert old.name not in copied
        assert env.transcriber.count == 1

    def test_a_clip_is_never_transcribed_twice(self, env) -> None:
        clip = env.clip("mic", BASE + 10)
        manager = env.manager()
        manager.start(env.meeting)
        env.wait_running(manager)
        env.clock.advance(30)

        wait_for(lambda: env.transcriber.count == 1, "the clip was never transcribed")
        env.wait_scans(manager, count=5)
        assert env.transcriber.count == 1, "a later scan resubmitted a clip it had already done"
        assert clip.name in state_of(env.meeting)["seen"]

    def test_a_clip_is_not_redone_after_a_catch_up_restart(self, env) -> None:
        env.clip("mic", BASE + 10)
        manager = env.manager()
        manager.start(env.meeting)
        env.wait_running(manager)
        env.clock.advance(30)
        wait_for(lambda: env.transcriber.count == 1, "the clip was never transcribed")
        manager.stop(env.meeting)

        resumed = env.manager()
        resumed.start(env.meeting, mode="catch_up")
        env.wait_running(resumed)
        env.wait_scans(resumed, count=5)

        assert env.transcriber.count == 1, "catch_up repeated a clip already marked seen"
        assert resumed.status(env.meeting)["lines_total"] == 1

    def test_restarting_with_mode_now_skips_clips_from_the_pause(self, env) -> None:
        env.clip("mic", BASE + 10)
        manager = env.manager()
        manager.start(env.meeting)
        env.wait_running(manager)
        env.clock.advance(30)
        wait_for(lambda: env.transcriber.count == 1, "the first clip was never transcribed")
        manager.stop(env.meeting)

        env.clip("system", BASE + 50)  # recorded while live was off
        env.clock.advance(100)

        resumed = env.manager()
        status = resumed.start(env.meeting, mode="now")
        assert status["watermark"] == env.clock.now
        env.wait_running(resumed)
        env.wait_scans(resumed, count=5)

        assert env.transcriber.clips == ["mic"], "mode=now transcribed a clip from before the restart"

    def test_restarting_with_mode_catch_up_picks_up_clips_from_the_pause(self, env) -> None:
        env.clip("mic", BASE + 10)
        manager = env.manager()
        manager.start(env.meeting)
        env.wait_running(manager)
        env.clock.advance(30)
        wait_for(lambda: env.transcriber.count == 1, "the first clip was never transcribed")
        manager.stop(env.meeting)

        env.clip("system", BASE + 50)
        env.clock.advance(100)

        resumed = env.manager()
        status = resumed.start(env.meeting, mode="catch_up")
        assert status["watermark"] == BASE, "catch_up must keep the saved watermark"
        env.wait_running(resumed)

        wait_for(
            lambda: sorted(env.transcriber.clips) == ["mic", "system"],
            "catch_up never picked up the clip recorded during the pause",
        )

    def test_a_clip_younger_than_the_age_threshold_waits(self, env) -> None:
        env.clip("mic", BASE + 10)
        manager = env.manager()
        manager.start(env.meeting)
        env.wait_running(manager)

        env.clock.advance(10.5)  # the clip is 0.5 s old: still being written
        env.wait_scans(manager, count=5)
        assert env.transcriber.count == 0, "a clip that may still be growing was submitted"
        assert not list(env.meeting.live_clips_dir.glob("*.wav"))

        env.clock.advance(10)
        wait_for(lambda: env.transcriber.count == 1, "the clip was never picked up once it settled")

    def test_a_truncated_clip_is_retried_not_marked_seen(self, env) -> None:
        clip = env.clip("mic", BASE + 10, seconds=1.0)
        truncate(clip)

        manager = env.manager()
        manager.start(env.meeting)
        env.wait_running(manager)
        env.clock.advance(30)
        env.wait_scans(manager, count=5)

        assert env.transcriber.count == 0, "a clip shorter than its own header was transcribed"
        assert state_of(env.meeting)["seen"] == [], "an unfinished clip must not be marked seen"

        # The producer finishes writing it; the next scan finds it consistent.
        write_wav(clip, tone(1.0), 16000)
        os.utime(clip, (BASE + 10, BASE + 10))
        wait_for(lambda: env.transcriber.count == 1, "the finished clip was never retried")

    def test_continuous_and_pcm_files_are_ignored(self, env) -> None:
        env.meeting.continuous_dir.mkdir(parents=True, exist_ok=True)
        session_file = env.meeting.continuous_dir / continuous_filename(BASE + 10, "system")
        write_wav(session_file, tone(1.0), 16000)
        os.utime(session_file, (BASE + 10, BASE + 10))

        stray = env.meeting.recordings_dir / continuous_filename(BASE + 11, "mic")
        write_wav(stray, tone(1.0), 16000)
        os.utime(stray, (BASE + 11, BASE + 11))

        raw = env.meeting.recordings_dir / "2026-08-24T19-28-51-917Z-mic-1001884403.pcm"
        raw.write_bytes(b"\x00" * 4096)
        os.utime(raw, (BASE + 12, BASE + 12))

        env.clip("mic", BASE + 20)

        manager = env.manager()
        manager.start(env.meeting)
        env.wait_running(manager)
        env.clock.advance(60)
        wait_for(lambda: env.transcriber.count == 1, "the real clip was never transcribed")
        env.wait_scans(manager, count=5)

        assert env.transcriber.count == 1
        copied = [path.name for path in env.meeting.live_clips_dir.iterdir()]
        assert all("continuous" not in name and not name.endswith(".pcm") for name in copied)

    def test_recordings_are_untouched_and_the_copies_live_elsewhere(self, env) -> None:
        clip = env.clip("mic", BASE + 10)
        before = fingerprint(env.meeting.recordings_dir)

        manager = env.manager()
        manager.start(env.meeting)
        env.wait_running(manager)
        env.clock.advance(30)
        wait_for(lambda: env.transcriber.count == 1, "the clip was never transcribed")
        manager.stop(env.meeting)

        assert fingerprint(env.meeting.recordings_dir) == before
        assert list(before) == [clip.name]
        copy = env.meeting.live_clips_dir / clip.name
        assert copy.exists() and sha256(copy) == sha256(clip)
        assert collect_wav_clips(env.meeting.recordings_dir) == [clip]


class TestTranscriptAndStatus:
    def test_one_line_per_utterance_in_the_documented_format(self, env) -> None:
        env.clip("mic", BASE + 10)
        env.clip("system", BASE + 20)

        manager = env.manager()
        manager.start(env.meeting)
        env.wait_running(manager)
        env.clock.advance(60)
        wait_for(lambda: manager.status(env.meeting)["lines_total"] == 2, "both clips never arrived")

        written = env.meeting.live_transcript_path.read_text(encoding="utf-8").splitlines()
        assert len(written) == 2
        for line in written:
            assert line.startswith("[") and "] " in line
        assert sorted(line.split("] ", 1)[1] for line in written) == [
            "mic: text from mic",
            "system: text from system",
        ]

    def test_after_returns_only_newer_lines(self, env) -> None:
        env.clip("mic", BASE + 10)
        env.clip("system", BASE + 20)
        manager = env.manager()
        manager.start(env.meeting)
        env.wait_running(manager)
        env.clock.advance(60)
        wait_for(lambda: manager.status(env.meeting)["lines_total"] == 2, "both clips never arrived")

        everything = manager.status(env.meeting)["lines"]
        assert [line["index"] for line in everything] == [0, 1]
        assert [line["index"] for line in manager.status(env.meeting, after=1)["lines"]] == [1]
        assert manager.status(env.meeting, after=2)["lines"] == []
        assert manager.status(env.meeting, after=2)["lines_total"] == 2

    def test_status_after_a_server_restart_comes_from_the_file(self, env) -> None:
        env.clip("mic", BASE + 10)
        manager = env.manager()
        manager.start(env.meeting, model_size="small")
        env.wait_running(manager)
        env.clock.advance(30)
        wait_for(lambda: manager.status(env.meeting)["lines_total"] == 1, "the clip never arrived")
        manager.stop(env.meeting)

        restarted = env.manager()  # a fresh process would know nothing
        status = restarted.status(env.meeting)

        assert set(status) == CONTRACT_KEYS
        assert status["status"] == "stopped"
        assert status["active"] is False
        assert status["mode"] is None
        assert status["model_size"] == "small", "the last session's model must survive a restart"
        assert status["watermark"] == BASE
        assert status["clips_seen"] == 1
        assert status["lines_total"] == 1
        assert status["lines"][0]["source"] == "mic"
        assert status["lines"][0]["text"] == "text from mic"
        assert status["lines"][0]["latency"] == 0.0
        assert status["lines"][0]["time"] == clock(BASE + 10), "the clock survives via the file, not started_at"
        assert restarted.global_status() == {"active": False, "meeting": None, "status": "stopped"}

    def test_lines_keep_their_index_across_a_restart(self, env) -> None:
        env.clip("mic", BASE + 10)
        manager = env.manager()
        manager.start(env.meeting)
        env.wait_running(manager)
        env.clock.advance(30)
        wait_for(lambda: manager.status(env.meeting)["lines_total"] == 1, "the clip never arrived")
        manager.stop(env.meeting)

        env.clip("system", BASE + 50)
        env.clock.advance(30)
        resumed = env.manager()
        resumed.start(env.meeting, mode="catch_up")
        env.wait_running(resumed)
        wait_for(lambda: resumed.status(env.meeting)["lines_total"] == 2, "the second clip never arrived")

        assert [line["index"] for line in resumed.status(env.meeting)["lines"]] == [0, 1]
        assert [line["index"] for line in resumed.status(env.meeting, after=1)["lines"]] == [1]

    def test_model_options_always_include_the_meetings_own_model(self, env) -> None:
        env.meeting.update_config({"model_size": "distil-large-v3"})
        status = env.manager().status(env.meeting)
        assert status["default_model"] == "distil-large-v3"
        assert status["model_options"] == MODEL_OPTIONS + ["distil-large-v3"]

    def test_transcript_text_is_empty_before_anything_ran(self, env) -> None:
        assert env.manager().transcript_text(env.meeting) == ""

    def test_parse_transcript_keeps_colons_in_the_text(self, tmp_path: Path) -> None:
        path = tmp_path / "live-transcript.txt"
        path.write_text("[12:30:45] mic: rule one: never block the recorder\n", encoding="utf-8")
        [line] = parse_transcript(path)
        assert line["time"] == "12:30:45"
        assert line["source"] == "mic"
        assert line["text"] == "rule one: never block the recorder"


class TestManagerLifecycle:
    def test_only_one_session_runs_at_a_time(self, env) -> None:
        manager = env.manager()
        manager.start(env.meeting)
        env.wait_running(manager)

        with pytest.raises(LiveBusy):
            manager.start(env.other)
        with pytest.raises(LiveBusy):
            manager.start(env.meeting)

        assert manager.global_status() == {
            "active": True,
            "meeting": f"{env.project.slug}/{env.meeting.slug}",
            "status": "running",
        }
        assert manager.status(env.other)["status"] == "stopped"

        manager.stop(env.meeting)
        manager.start(env.other)  # the slot is free again
        env.wait_running(manager, env.other)

    def test_stopping_something_that_is_not_running_is_a_no_op(self, env) -> None:
        manager = env.manager()
        assert manager.stop(env.meeting)["status"] == "stopped"

        manager.start(env.meeting)
        env.wait_running(manager)
        assert manager.stop(env.other)["status"] == "stopped", "stopping another meeting must not touch this one"
        assert manager.status(env.meeting)["active"] is True
        assert manager.stop(env.meeting)["active"] is False
        assert manager.stop(env.meeting)["active"] is False

    def test_a_model_that_cannot_load_reports_the_reason_and_frees_the_slot(self, env) -> None:
        def explode(config: dict):
            raise RuntimeError("no CUDA device")

        manager = env.manager(factory=explode)
        manager.start(env.meeting)
        wait_for(lambda: manager.status(env.meeting)["status"] == "error", "the failed load never surfaced")

        status = manager.status(env.meeting)
        assert status["error"] == "no CUDA device"
        assert status["active"] is False
        assert manager.global_status()["active"] is False

        working = env.manager()
        working.start(env.meeting)  # must not be blocked by the dead session
        env.wait_running(working)

    def test_an_unknown_mode_is_refused(self, env) -> None:
        with pytest.raises(ValueError):
            env.manager().start(env.meeting, mode="whenever")

    def test_the_fake_transcriber_needs_no_model(self) -> None:
        from avilistener.audio import AudioChunk

        fake = FakeLiveTranscriber({}, delay=0.0)
        chunk = AudioChunk(source="mic", audio=tone(0.2), sample_rate=16000, started_at=BASE, ended_at=BASE + 0.2, rms=0.1)
        assert fake.transcribe(chunk).text == "simulated transcript of mic clip"
        empty = AudioChunk(source="mic", audio=np.zeros(0, dtype=np.float32), sample_rate=16000, started_at=BASE, ended_at=BASE, rms=0.0)
        assert fake.transcribe(empty) is None


class TestLiveFilesDoNotDisturbTheOfflinePipeline:
    """`live/` is a sibling of `recordings/` so nothing derived can see it."""

    def test_meeting_views_are_unchanged_by_live_output(self, env) -> None:
        env.clip("mic", BASE + 10)
        env.meeting.continuous_dir.mkdir(parents=True, exist_ok=True)
        (env.meeting.continuous_dir / continuous_filename(BASE, "system")).write_bytes(b"")

        before = (
            env.meeting.segment_counts(),
            env.meeting.artifacts(),
            [path.name for path in env.meeting.continuous_parts("system")],
            [path.name for path in env.meeting.shared_parts()],
            [path.name for path in collect_wav_clips(env.meeting.recordings_dir)],
        )

        manager = env.manager()
        manager.start(env.meeting)
        env.wait_running(manager)
        env.clock.advance(30)
        wait_for(lambda: manager.status(env.meeting)["lines_total"] == 1, "the clip never arrived")
        manager.stop(env.meeting)

        assert env.meeting.live_clips_dir.exists(), "the session should have written under live/"
        assert env.meeting.live_state_path.exists()
        after = (
            env.meeting.segment_counts(),
            env.meeting.artifacts(),
            [path.name for path in env.meeting.continuous_parts("system")],
            [path.name for path in env.meeting.shared_parts()],
            [path.name for path in collect_wav_clips(env.meeting.recordings_dir)],
        )
        assert after == before
        assert before[0] == {"mic": 1}

    def test_a_stale_transcript_check_ignores_live_copies(self, env) -> None:
        clip = env.clip("mic", BASE + 10)
        env.meeting.transcripts_dir.mkdir(parents=True, exist_ok=True)
        (env.meeting.transcripts_dir / "inputs.json").write_text(
            json.dumps([{"filename": clip.name, "bytes": clip.stat().st_size}]), encoding="utf-8"
        )
        assert env.meeting.artifacts()["transcription_stale"] is False

        env.meeting.live_clips_dir.mkdir(parents=True, exist_ok=True)
        (env.meeting.live_clips_dir / clip.name).write_bytes(clip.read_bytes())
        assert env.meeting.artifacts()["transcription_stale"] is False


class TestRoutes:
    """The route functions are called directly; this repo keeps no TestClient."""

    @pytest.fixture()
    def routes(self, env, monkeypatch):
        monkeypatch.setattr(app_module, "workspace", env.workspace)
        manager = env.manager()
        monkeypatch.setattr(app_module, "live", manager)
        return manager

    def test_get_returns_the_contract_shape(self, env, routes) -> None:
        payload = app_module.live_status(env.project.slug, env.meeting.slug)
        assert set(payload) == CONTRACT_KEYS
        assert payload["status"] == "stopped"
        assert payload["active"] is False
        assert payload["lines"] == []
        assert payload["model_options"] == MODEL_OPTIONS

    def test_start_and_stop_return_the_same_shape(self, env, routes) -> None:
        env.clip("mic", BASE + 10)
        started = app_module.live_start(
            env.project.slug, env.meeting.slug, app_module.LiveStartIn(model_size="tiny", mode="now")
        )
        assert set(started) == CONTRACT_KEYS
        assert started["active"] is True
        assert started["model_size"] == "tiny"
        assert started["mode"] == "now"
        assert started["lines"] == [], "start does not ship lines"

        env.wait_running(routes)
        env.clock.advance(30)
        wait_for(
            lambda: app_module.live_status(env.project.slug, env.meeting.slug)["lines_total"] == 1,
            "the clip never arrived through the route",
        )

        assert app_module.live_status_global() == {
            "active": True,
            "meeting": f"{env.project.slug}/{env.meeting.slug}",
            "status": "running",
        }

        stopped = app_module.live_stop(env.project.slug, env.meeting.slug)
        assert set(stopped) == CONTRACT_KEYS
        assert stopped["status"] == "stopped"
        assert stopped["active"] is False
        assert stopped["lines_total"] == 1
        assert app_module.live_status_global()["meeting"] is None

    def test_transcript_is_plain_text(self, env, routes) -> None:
        env.meeting.live_transcript_path.parent.mkdir(parents=True, exist_ok=True)
        env.meeting.live_transcript_path.write_text("[12:00:00] mic: hello\n", encoding="utf-8")
        response = app_module.live_transcript(env.project.slug, env.meeting.slug)
        assert response.media_type == "text/plain"
        assert response.body.decode("utf-8") == "[12:00:00] mic: hello\n"

    def test_transcript_is_empty_when_nothing_ran(self, env, routes) -> None:
        assert app_module.live_transcript(env.project.slug, env.meeting.slug).body == b""

    def test_the_meeting_payload_carries_live_without_lines(self, env, routes) -> None:
        payload = app_module.get_meeting(env.project.slug, env.meeting.slug)
        assert set(payload["live"]) == CONTRACT_KEYS
        assert payload["live"]["lines"] == []

    def test_a_second_meeting_cannot_start_while_one_runs(self, env, routes) -> None:
        app_module.live_start(env.project.slug, env.meeting.slug, app_module.LiveStartIn())
        env.wait_running(routes)

        with pytest.raises(HTTPException) as raised:
            app_module.live_start(env.project.slug, env.other.slug, app_module.LiveStartIn())
        assert raised.value.status_code == 409

    def test_a_bad_mode_is_a_400(self, env, routes) -> None:
        with pytest.raises(HTTPException) as raised:
            app_module.live_start(
                env.project.slug, env.meeting.slug, app_module.LiveStartIn(mode="sometime")
            )
        assert raised.value.status_code == 400

    def test_an_unknown_meeting_is_a_404(self, env, routes) -> None:
        for call in (
            lambda: app_module.live_status(env.project.slug, "nope"),
            lambda: app_module.live_start(env.project.slug, "nope", app_module.LiveStartIn()),
            lambda: app_module.live_stop(env.project.slug, "nope"),
            lambda: app_module.live_transcript(env.project.slug, "nope"),
            lambda: app_module.live_status("nope", env.meeting.slug),
        ):
            with pytest.raises(HTTPException) as raised:
                call()
            assert raised.value.status_code == 404
