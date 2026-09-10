from __future__ import annotations

import threading
import wave

import numpy as np
import pytest

from avilistener.server.jobs import RecordingManager, RecordingSession


def pcm(seconds: float, amplitude: float, sample_rate: int = 16000) -> bytes:
    frames = int(seconds * sample_rate)
    time = np.arange(frames, dtype=np.float32) / sample_rate
    audio = np.sin(2 * np.pi * 440 * time) * amplitude
    return audio.astype("<f4").tobytes()


def test_manager_records_a_browser_only_session_without_audio_hardware(tmp_path) -> None:
    manager = RecordingManager()
    status = manager.start(
        meeting_key="project/meeting",
        output_dir=tmp_path / "recordings",
        sources=["chrome"],
        sample_rate=16000,
        devices={},
        log_dir=tmp_path / "logs",
    )

    assert status["running"] is True
    assert status["sources"] == ["chrome"]
    assert status["continuous_files"]["chrome"] == {
        "filename": None,
        "bytes": 0,
        "state": "waiting",
    }
    assert manager.push_browser_audio("project/meeting", pcm(0.5, 0.4), 16000) == 8000

    growing = manager.status()["continuous_files"]["chrome"]
    assert growing["filename"].endswith("-chrome-continuous.wav")
    assert growing["bytes"] == 44 + 8000 * 2
    assert growing["state"] == "growing"

    assert manager.push_browser_audio("project/meeting", pcm(2.0, 0.0), 16000) == 32000

    stopped = manager.stop()
    assert stopped["running"] is False
    assert stopped["segments"]["chrome"] == 1
    assert stopped["continuous_files"]["chrome"]["state"] == "complete"
    [continuous] = list((tmp_path / "recordings" / "continuous").glob("*-chrome-continuous.wav"))
    with wave.open(str(continuous), "rb") as handle:
        assert handle.getframerate() == 16000
        assert handle.getnframes() == 40000


def test_manager_rejects_browser_chunks_when_chrome_was_not_selected(tmp_path, monkeypatch) -> None:
    manager = RecordingManager()

    class FakeDeviceRecorder:
        source = type("Source", (), {"name": "mic"})()
        saved = 0
        peak_level = 0.0
        continuous_seconds = 0.0
        continuous_path = None
        error = None

        def __init__(self, *args, **kwargs):
            self.running = False

        def start(self):
            self.running = True

        def is_alive(self):
            return self.running

        def join(self, timeout=None):
            self.running = False

    monkeypatch.setattr("avilistener.server.jobs.SourceRecorder", FakeDeviceRecorder)
    manager.start("project/meeting", tmp_path, ["mic"], 16000, {})
    with pytest.raises(RuntimeError, match="not selected"):
        manager.push_browser_audio("project/meeting", pcm(0.1, 0.2), 16000)
    manager.stop()


def test_manager_rejects_browser_chunks_for_another_meeting(tmp_path) -> None:
    manager = RecordingManager()
    manager.start("project/meeting", tmp_path, ["chrome"], 16000, {})
    with pytest.raises(RuntimeError, match="not recording"):
        manager.push_browser_audio("project/other", pcm(0.1, 0.2), 16000)
    manager.stop()


def test_manager_rejects_duplicate_pc_and_tab_capture(tmp_path) -> None:
    manager = RecordingManager()
    with pytest.raises(ValueError, match="either PC audio or Chrome tab"):
        manager.start("project/meeting", tmp_path, ["system", "chrome"], 16000, {})


def test_recording_status_marks_a_continuous_file_stalled_until_it_grows(tmp_path, monkeypatch) -> None:
    now = [100.0]
    monkeypatch.setattr("avilistener.server.jobs.time.time", lambda: now[0])

    class FakeRecorder:
        source = type("Source", (), {"name": "mic"})()
        continuous_path = tmp_path / "continuous" / "mic.wav"
        saved = 0
        peak_level = 0.0
        error = None
        alive = True

        def is_alive(self):
            return self.alive

    recorder = FakeRecorder()
    session = RecordingSession(
        meeting="project/meeting",
        output_dir=tmp_path,
        sources=["mic"],
        started_at=now[0],
        stop_event=threading.Event(),
        recorders=[recorder],
    )

    assert session.status()["continuous_files"]["mic"]["state"] == "waiting"

    now[0] += RecordingSession.STALL_AFTER_SECONDS
    assert session.status()["continuous_files"]["mic"]["state"] == "stalled"

    recorder.continuous_path.parent.mkdir()
    recorder.continuous_path.write_bytes(b"audio is arriving")
    growing = session.status()["continuous_files"]["mic"]
    assert growing["bytes"] == len(b"audio is arriving")
    assert growing["state"] == "growing"

    now[0] += RecordingSession.STALL_AFTER_SECONDS
    assert session.status()["continuous_files"]["mic"]["state"] == "stalled"


def test_recording_status_prioritizes_capture_errors_over_file_growth(tmp_path) -> None:
    class FailedRecorder:
        source = type("Source", (), {"name": "mic"})()
        continuous_path = tmp_path / "mic.wav"
        saved = 0
        peak_level = 0.0
        error = RuntimeError("device disconnected")

        def is_alive(self):
            return False

    session = RecordingSession(
        meeting="project/meeting",
        output_dir=tmp_path,
        sources=["mic"],
        started_at=0.0,
        stop_event=threading.Event(),
        recorders=[FailedRecorder()],
    )

    assert session.status()["continuous_files"]["mic"]["state"] == "failed"
