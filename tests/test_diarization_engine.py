"""Choosing between diarization engines."""

from __future__ import annotations

from pathlib import Path

import pytest

from avilistener.meeting import run_meeting_diarize, run_pyannote
from avilistener.server.workspace import DEFAULT_PROJECT_CONFIG


def test_unknown_engine_fails_before_touching_anything(tmp_path):
    with pytest.raises(SystemExit, match="whisper-x"):
        run_meeting_diarize(
            audio_path=tmp_path / "a.wav",
            config={},
            output_dir=tmp_path / "out",
            num_speakers=None,
            max_speakers=6,
            engine="whisper-x",
        )
    assert not (tmp_path / "out").exists()


def test_pyannote_missing_venv_says_how_to_fix_it(tmp_path):
    with pytest.raises(SystemExit, match=r"\.venv-pyannote"):
        run_pyannote(tmp_path / "a.wav", tmp_path, None, 6,
                     pyannote_python=tmp_path / "nowhere" / "python.exe")


def test_projects_default_to_pyannote():
    # The measured better engine is the default; a missing Hugging Face token
    # is caught before the job starts, with instructions instead of a crash.
    assert DEFAULT_PROJECT_CONFIG["diarization_engine"] == "pyannote"
