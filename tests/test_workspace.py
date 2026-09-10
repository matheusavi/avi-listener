"""Projects, meetings, and which step the dashboard may offer next.

The interface greys out steps using `capabilities()`, so a wrong answer here
either hides work the user can do or offers one that will fail.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from avilistener.server.workspace import Workspace, slugify


@pytest.fixture()
def workspace(tmp_path: Path) -> Workspace:
    return Workspace(root=tmp_path / "workspace")


def make_meeting(workspace: Workspace, project_name: str = "Work", meeting_name: str = "Standup"):
    project = workspace.create_project(project_name)
    return project, project.create_meeting(meeting_name)


def add_segment(meeting, source: str) -> None:
    meeting.recordings_dir.mkdir(parents=True, exist_ok=True)
    (meeting.recordings_dir / f"2026-08-24T19-28-51-917Z-{source}-1001884403.wav").write_bytes(b"")


def add_continuous(meeting, source: str) -> None:
    meeting.continuous_dir.mkdir(parents=True, exist_ok=True)
    (meeting.continuous_dir / f"2026-08-24T19-28-50-097Z-{source}-continuous.wav").write_bytes(b"")


class TestSlugify:
    def test_makes_names_safe_for_the_filesystem(self) -> None:
        assert slugify("Meet: Q3 Planning / draft") == "meet-q3-planning-draft"

    def test_never_returns_empty(self) -> None:
        assert slugify("   ") == "untitled"
        assert slugify("///") == "untitled"


class TestProjectsAndMeetings:
    def test_meetings_are_separate_folders(self, workspace: Workspace) -> None:
        project, first = make_meeting(workspace)
        second = project.create_meeting("Retro")
        assert first.path != second.path
        assert {meeting.slug for meeting in project.meetings()} == {first.slug, second.slug}

    def test_two_meetings_with_the_same_name_do_not_collide(self, workspace: Workspace) -> None:
        """Otherwise the second session would record into the first one's folder."""
        project = workspace.create_project("Work")
        first = project.create_meeting("Standup")
        second = project.create_meeting("Standup")
        assert first.slug != second.slug
        assert first.name == second.name == "Standup"

    def test_meeting_inherits_project_config(self, workspace: Workspace) -> None:
        project = workspace.create_project("Work", {"language": "pt", "me": "Alice"})
        meeting = project.create_meeting("Standup")
        assert meeting.config["language"] == "pt"
        assert meeting.config["me"] == "Alice"

    def test_meeting_can_override_the_project_without_affecting_siblings(self, workspace: Workspace) -> None:
        project = workspace.create_project("Work", {"language": "pt"})
        first = project.create_meeting("English call")
        second = project.create_meeting("Normal call")
        first.update_config({"language": "en"})

        assert first.config["language"] == "en"
        assert second.config["language"] == "pt"
        assert project.config["language"] == "pt"

    def test_config_survives_a_reload_from_disk(self, workspace: Workspace) -> None:
        project, meeting = make_meeting(workspace)
        meeting.update_config({"num_speakers": 3})

        reopened = Workspace(root=workspace.root).project(project.slug).meeting(meeting.slug)
        assert reopened.config["num_speakers"] == 3

    def test_unknown_project_or_meeting_is_none(self, workspace: Workspace) -> None:
        assert workspace.project("nope") is None
        project = workspace.create_project("Work")
        assert project.meeting("nope") is None


class TestSecrets:
    """The Discord token is stored but must never travel back to the browser."""

    def test_token_is_persisted(self, workspace: Workspace) -> None:
        project = workspace.create_project("Work")
        project.update_config({"discord_token": "s3cret-token"})

        reopened = Workspace(root=workspace.root).project(project.slug)
        assert reopened.config["discord_token"] == "s3cret-token"

    def test_token_is_stripped_from_api_output(self, workspace: Workspace) -> None:
        project = workspace.create_project("Work")
        project.update_config({"discord_token": "s3cret-token"})
        meeting = project.create_meeting("Standup")

        for payload in (project.to_json(), meeting.to_json()):
            serialized = json.dumps(payload)
            assert "s3cret-token" not in serialized
            assert "discord_token" not in payload["config"]
            assert payload["config"]["discord_token_set"] is True

    def test_absent_token_reports_not_set(self, workspace: Workspace) -> None:
        project = workspace.create_project("Work")
        assert project.to_json()["config"]["discord_token_set"] is False

    def test_blank_token_counts_as_not_set(self, workspace: Workspace) -> None:
        project = workspace.create_project("Work")
        project.update_config({"discord_token": "   "})
        assert project.to_json()["config"]["discord_token_set"] is False

    def test_meeting_inherits_the_projects_token(self, workspace: Workspace) -> None:
        """Recording reads it from the meeting, so inheritance has to hold."""
        project = workspace.create_project("Work")
        project.update_config({"discord_token": "s3cret-token"})
        meeting = project.create_meeting("Standup")
        assert meeting.config["discord_token"] == "s3cret-token"


class TestCapabilities:
    def test_nothing_is_offered_before_recording(self, workspace: Workspace) -> None:
        _, meeting = make_meeting(workspace)
        assert meeting.capabilities() == {"transcribe": False, "diarize": False, "combine": False}

    def test_recording_the_microphone_alone_does_not_enable_diarization(self, workspace: Workspace) -> None:
        """A microphone holds one person; splitting it invents speakers."""
        _, meeting = make_meeting(workspace)
        add_segment(meeting, "mic")
        add_continuous(meeting, "mic")

        assert meeting.capabilities()["transcribe"] is True
        assert meeting.capabilities()["diarize"] is False

    def test_loopback_enables_diarization(self, workspace: Workspace) -> None:
        _, meeting = make_meeting(workspace)
        add_continuous(meeting, "system")
        assert meeting.capabilities()["diarize"] is True
        assert meeting.artifacts()["loopback"].endswith("-system-continuous.wav")

    def test_chrome_tab_audio_enables_diarization(self, workspace: Workspace) -> None:
        _, meeting = make_meeting(workspace)
        add_continuous(meeting, "chrome")
        assert meeting.capabilities()["diarize"] is True
        assert meeting.artifacts()["continuous"]["chrome"] is True
        assert meeting.artifacts()["loopback"].endswith("-chrome-continuous.wav")

    def test_combine_needs_both_halves(self, workspace: Workspace) -> None:
        _, meeting = make_meeting(workspace)
        add_continuous(meeting, "system")
        meeting.diarized_dir.mkdir(parents=True, exist_ok=True)
        (meeting.diarized_dir / "events.json").write_text("[]", encoding="utf-8")
        assert meeting.capabilities()["combine"] is False, "diarization alone is not enough"

        meeting.transcripts_dir.mkdir(parents=True, exist_ok=True)
        (meeting.transcripts_dir / "events.jsonl").write_text("", encoding="utf-8")
        assert meeting.capabilities()["combine"] is True

    def test_segments_are_counted_per_source_including_processed(self, workspace: Workspace) -> None:
        _, meeting = make_meeting(workspace)
        add_segment(meeting, "mic")
        add_segment(meeting, "system")
        processed = meeting.recordings_dir / "processed"
        processed.mkdir(parents=True, exist_ok=True)
        (processed / "2026-08-24T19-29-01-917Z-mic-1001884403.wav").write_bytes(b"")

        # Transcribing moves files aside; the count must not appear to drop.
        assert meeting.segment_counts() == {"mic": 2, "system": 1}

    def test_to_json_exposes_what_the_interface_needs(self, workspace: Workspace) -> None:
        _, meeting = make_meeting(workspace)
        payload = json.loads(json.dumps(meeting.to_json()))
        assert {"slug", "name", "config", "artifacts", "can"} <= set(payload)
