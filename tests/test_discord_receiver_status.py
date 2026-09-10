"""A Discord-only recording has no local audio session, so the interface
relies on the receiver's own status to know that it is recording, and the
receiver is one process shared by every meeting."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from avilistener.server import app as app_module
from avilistener.server.jobs import DiscordReceiver
from avilistener.server.workspace import Workspace


class _AliveProcess:
    def poll(self):
        return None


@pytest.fixture()
def meetings(tmp_path: Path):
    workspace = Workspace(root=tmp_path / "workspace")
    project = workspace.create_project("Guild")
    return project.create_meeting("Raid night"), project.create_meeting("Lounge chat")


def test_status_reports_meeting_and_elapsed_while_running(tmp_path: Path) -> None:
    receiver = DiscordReceiver(project_root=tmp_path)
    receiver.process = _AliveProcess()
    receiver.meeting = "guild/raid-night"
    receiver.started_at = time.time() - 12

    status = receiver.status()

    assert status["running"] is True
    assert status["meeting"] == "guild/raid-night"
    assert status["elapsed"] >= 12


def test_status_when_idle_has_no_elapsed(tmp_path: Path) -> None:
    receiver = DiscordReceiver(project_root=tmp_path)
    status = receiver.status()
    assert status["running"] is False
    assert status["elapsed"] is None


def test_another_meeting_does_not_see_the_recording(monkeypatch, meetings) -> None:
    raid, lounge = meetings

    class Stub:
        def status(self):
            return {"running": True, "meeting": app_module._key(raid), "started_at": 1.0, "elapsed": 5.0, "log": ["joined"]}

    monkeypatch.setattr(app_module, "discord", Stub())

    assert app_module._discord_status_for(raid)["running"] is True
    assert app_module._discord_status_for(raid)["log"] == ["joined"]
    other = app_module._discord_status_for(lounge)
    assert other["running"] is False
    assert other["log"] == []
