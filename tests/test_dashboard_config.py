"""The dashboard must transcribe with the same tuned settings as the CLI.

The first version passed an empty `transcription` block, so faster-whisper fell
back to its own defaults: no hallucination filtering, no hotwords, and looser
thresholds. Transcripts were quietly worse than the command line for the same
audio, with nothing to indicate why.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from avilistener.transcriber import build_transcriber
from avilistener.server.app import _transcription_config
from avilistener.server.workspace import DEFAULT_PROJECT_CONFIG, Workspace


@pytest.fixture()
def meeting(tmp_path: Path):
    workspace = Workspace(root=tmp_path / "workspace")
    project = workspace.create_project("Work")
    return project.create_meeting("Standup")


class TestTranscriptionConfig:
    def test_tuned_options_reach_the_transcriber(self, meeting) -> None:
        options = _transcription_config(meeting)["transcription"]
        assert options["beam_size"] == 8, "stock default is 5"
        assert options["best_of"] == 8
        assert options["no_repeat_ngram_size"] == 3

    def test_hallucination_phrases_are_configured(self, meeting) -> None:
        """Whisper invents subtitle credits over silence; they must be dropped."""
        phrases = _transcription_config(meeting)["transcription"]["ignored_phrases"]
        assert any("inscreva" in phrase for phrase in phrases)
        assert any("legendas por" in phrase for phrase in phrases)

    def test_vad_parameters_survive_as_a_nested_dict(self, meeting) -> None:
        vad = _transcription_config(meeting)["transcription"]["vad_parameters"]
        assert vad["min_silence_duration_ms"] == 350
        assert vad["speech_pad_ms"] == 250

    def test_a_project_can_override_the_word_lists(self, meeting) -> None:
        meeting.project.update_config(
            {"transcription": {**DEFAULT_PROJECT_CONFIG["transcription"], "hotwords": ["Camaro", "Pagani"]}}
        )
        assert _transcription_config(meeting)["transcription"]["hotwords"] == ["Camaro", "Pagani"]

    def test_a_meeting_override_does_not_leak_to_siblings(self, meeting) -> None:
        other = meeting.project.create_meeting("Other")
        meeting.update_config({"language": "en"})
        assert _transcription_config(meeting)["language"] == "en"
        assert _transcription_config(other)["language"] == "pt"

    def test_options_are_accepted_by_the_transcriber(self, meeting, monkeypatch) -> None:
        """Guards against a key the transcriber does not understand."""
        built = {}

        class FakeModel:
            def __init__(self, *args, **kwargs) -> None:
                built["model"] = kwargs

        monkeypatch.setitem(
            __import__("sys").modules,
            "faster_whisper",
            type("m", (), {"WhisperModel": FakeModel}),
        )
        config = _transcription_config(meeting)
        config["device"] = "cpu"  # avoid loading CUDA libraries in a test
        transcriber = build_transcriber(config)

        assert transcriber.options.beam_size == 8
        assert transcriber.options.hotwords is None or isinstance(transcriber.options.hotwords, str)
        assert "legendas por" in transcriber.options.ignored_phrases
