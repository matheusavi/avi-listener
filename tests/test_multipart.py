"""Restarted recordings must keep both halves and the real gap, without edits to inputs."""
import json
import wave
from pathlib import Path

import numpy as np
import pytest

from avilistener.combine import load_diarized_events, load_transcript_events
from avilistener.meeting import DiarizationTurn, _emit_speaker_lines
from avilistener.processing import combine_meeting, diarize_meeting, transcribe_meeting
from avilistener.recorder import continuous_filename, segment_filename
from avilistener.server.workspace import Workspace
from avilistener.timeline import build_timeline, inspect_parts, recording_manifest
from avilistener.transcriber import TranscriptResult
from avilistener.writer import TranscriptWriter

START = 1788822862.0


def wav(path, rate=16000, seconds=2, value=500, channels=1):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(channels)
        writer.setsampwidth(2)
        writer.setframerate(rate)
        writer.writeframes(np.full(round(rate * seconds) * channels, value, dtype="<i2").tobytes())
    return path


@pytest.fixture
def meeting(tmp_path):
    return Workspace(tmp_path).create_project("work").create_meeting("resumed")


def parts_for(meeting):
    return [
        wav(meeting.continuous_dir / continuous_filename(START, "chrome"), value=500),
        wav(meeting.continuous_dir / continuous_filename(START + 1000, "chrome"), value=-500),
    ]


def test_join_is_lossless_and_gap_is_only_in_mapping(meeting, tmp_path):
    paths = parts_for(meeting)
    original = recording_manifest(meeting.recordings_dir)
    parts = inspect_parts(paths)
    target = build_timeline(paths, tmp_path / "output", parts)
    with wave.open(str(target), "rb") as audio:
        samples = np.frombuffer(audio.readframes(audio.getnframes()), dtype="<i2")
    assert len(samples) == 64000
    assert np.all(samples[:32000] == 500)
    assert np.all(samples[32000:] == -500)
    assert parts[1]["offset"] == 2
    assert parts[1]["started_at"] - parts[0]["ended_at"] == 998
    assert recording_manifest(meeting.recordings_dir) == original


def test_join_handles_changed_browser_sample_rate(meeting, tmp_path):
    paths = parts_for(meeting)
    wav(paths[1], rate=48000, seconds=2.1, channels=2)
    parts = inspect_parts(paths)
    target = build_timeline(paths, tmp_path / "output", parts)
    with wave.open(str(target), "rb") as audio:
        assert audio.getframerate() == 16000
        assert audio.getnchannels() == 1
        assert audio.getnframes() == 65600
    assert parts[1]["duration"] == 2.1


def test_same_speaker_never_joins_lines_across_restart(meeting):
    paths = parts_for(meeting)
    parts = inspect_parts(paths)
    build_timeline(paths, meeting.diarized_dir, parts)
    _emit_speaker_lines([(0.2, 1.8, "first"), (2.2, 3.8, "second")], [],
                        [DiarizationTurn(0, 4, "speaker_0")], meeting.diarized_dir, {})
    events = load_diarized_events(meeting.diarized_dir, START)
    assert len(events) == 2
    assert events[0].text == "first"
    assert events[1].text == "second"
    assert events[0].started_at == pytest.approx(START + .2)
    assert events[1].started_at == pytest.approx(START + 1000.2)
    assert events[0].ended_at < START + 2


def test_merge_keeps_both_remote_parts_mic_and_uncovered_clips(meeting):
    paths = parts_for(meeting)
    parts = inspect_parts(paths)
    build_timeline(paths, meeting.diarized_dir, parts)
    _emit_speaker_lines([(0.2, 1.8, "remote first"), (2.2, 3.8, "remote second")], [],
                        [DiarizationTurn(0, 4, "speaker_0")], meeting.diarized_dir, {})
    writer = TranscriptWriter(meeting.transcripts_dir)
    for source, delta, text in [("mic", 1, "mic first"), ("mic", 1001, "mic second"),
                                ("chrome", 0, "duplicate first"), ("chrome", 1000, "duplicate second"),
                                ("chrome", 600, "uncovered remote")]:
        writer.write(TranscriptResult(source, text, START + delta, START + delta + .5, .1))
    combine_meeting(meeting)
    events = [json.loads(line) for line in (meeting.merged_dir / "events.jsonl").read_text().splitlines()]
    assert [e["text"] for e in events] == ["remote first", "mic first", "uncovered remote", "remote second", "mic second"]
    assert all("duplicate" not in e["text"] for e in events)


def test_partial_coverage_keeps_boundary_clips(tmp_path):
    writer = TranscriptWriter(tmp_path)
    writer.write(TranscriptResult("chrome", "spans boundary", 100, 104, .1))
    assert len(load_transcript_events(writer.events_path, "chrome", coverage=[{"source": "chrome", "started_at": 102, "ended_at": 105}])) == 1


def fake_engine(audio_path, config, output_dir, **kwargs):
    timeline = json.loads((output_dir / "timeline.json").read_text())
    total = sum(p["duration"] for p in timeline["parts"])
    (output_dir / "session-audio.rttm").write_text(f"SPEAKER session-audio 1 0 {total} <NA> <NA> speaker_0 <NA> <NA>\n")
    (output_dir / "diarization.json").write_text(json.dumps([{"start": 0, "end": total, "speaker": "speaker_0"}]))
    _emit_speaker_lines([(p["offset"] + .2, p["offset"] + 1, f"part {i}") for i, p in enumerate(timeline["parts"])],
                        [], [DiarizationTurn(0, total, "speaker_0")], output_dir, {})


def test_cache_only_reuses_identical_audio_and_settings(meeting, monkeypatch):
    parts_for(meeting)
    cached = []
    def engine(**kwargs):
        cached.append((kwargs["output_dir"] / "words.json").exists())
        fake_engine(**kwargs)
    monkeypatch.setattr("avilistener.meeting.run_meeting_diarize", engine)
    diarize_meeting(meeting)
    diarize_meeting(meeting, num_speakers=2)
    wav(meeting.continuous_dir / continuous_filename(START + 2000, "chrome"))
    diarize_meeting(meeting)
    meeting.update_config({"language": "en"})
    diarize_meeting(meeting)
    assert cached == [False, True, False, False]
    assert len(list((meeting.path / "artifact-history").glob("*-diarized-*"))) == 3
    assert meeting.speaker_audio().exists()


def test_failed_diarization_preserves_previous_results_and_originals(meeting, monkeypatch):
    parts_for(meeting)
    original = recording_manifest(meeting.recordings_dir)
    monkeypatch.setattr("avilistener.meeting.run_meeting_diarize", fake_engine)
    diarize_meeting(meeting)
    previous = (meeting.diarized_dir / "events.json").read_bytes()
    def fail(**kwargs):
        raise RuntimeError("GPU error")
    monkeypatch.setattr("avilistener.meeting.run_meeting_diarize", fail)
    with pytest.raises(RuntimeError, match="GPU error"):
        diarize_meeting(meeting)
    assert (meeting.diarized_dir / "events.json").read_bytes() == previous
    assert recording_manifest(meeting.recordings_dir) == original


def test_transcribe_again_includes_processed_and_keeps_originals(meeting, monkeypatch):
    a = wav(meeting.recordings_dir / segment_filename(START, "mic"))
    b = wav(meeting.recordings_dir / "processed" / segment_filename(START + 1000, "mic"))
    original = recording_manifest(meeting.recordings_dir)
    calls = []
    class Model:
        def transcribe(self, chunk):
            calls.append(chunk.started_at)
            return TranscriptResult(chunk.source, "speech", chunk.started_at, chunk.ended_at, chunk.rms)
    monkeypatch.setattr("avilistener.transcriber.build_transcriber", lambda c: Model())
    transcribe_meeting(meeting)
    transcribe_meeting(meeting)
    assert len(calls) == 2  # cache makes re-runs repeatable without a second GPU pass
    assert len((meeting.transcripts_dir / "events.jsonl").read_text().splitlines()) == 2
    assert recording_manifest(meeting.recordings_dir) == original
    assert a.exists() and b.exists()


def test_duplicate_legacy_clip_is_read_once_and_conflicting_copy_is_rejected(meeting, monkeypatch):
    from avilistener.file_transcriber import collect_wav_clips
    import shutil
    original = wav(meeting.recordings_dir / segment_filename(START, "mic"))
    copy = meeting.recordings_dir / "processed" / original.name
    copy.parent.mkdir()
    shutil.copy2(original, copy)
    assert collect_wav_clips(meeting.recordings_dir) == [original]
    wav(copy, value=1000)
    with pytest.raises(ValueError, match="Conflicting"):
        collect_wav_clips(meeting.recordings_dir)


def test_failed_transcription_keeps_previous_output(meeting, monkeypatch):
    wav(meeting.recordings_dir / segment_filename(START, "mic"))
    meeting.transcripts_dir.mkdir()
    previous = meeting.transcripts_dir / "combined.txt"
    previous.write_text("previous successful transcript")
    class BrokenModel:
        def transcribe(self, chunk):
            raise RuntimeError("CUDA failed")
    monkeypatch.setattr("avilistener.transcriber.build_transcriber", lambda config: BrokenModel())
    with pytest.raises(RuntimeError, match="CUDA failed"):
        transcribe_meeting(meeting)
    assert previous.read_text() == "previous successful transcript"


def test_invalid_part_fails_without_modifying_it(meeting):
    paths = parts_for(meeting)
    paths[1].write_bytes(b"broken header")
    before = paths[1].read_bytes()
    with pytest.raises((wave.Error, EOFError)):
        inspect_parts(paths)
    assert paths[1].read_bytes() == before


def test_shared_parts_include_source_switches(meeting):
    paths = parts_for(meeting)
    third = wav(meeting.continuous_dir / continuous_filename(START + 2000, "system"))
    assert meeting.shared_parts() == paths + [third]
    assert meeting.artifacts()["continuous_parts"]["chrome"] == 2


def test_resuming_after_processing_blocks_stale_merge_until_all_parts_are_processed(meeting, monkeypatch):
    parts_for(meeting)
    monkeypatch.setattr("avilistener.meeting.run_meeting_diarize", fake_engine)
    diarize_meeting(meeting)
    writer = TranscriptWriter(meeting.transcripts_dir)
    writer.write(TranscriptResult("mic", "original", START, START + 1, .1))
    assert meeting.capabilities()["combine"]
    wav(meeting.continuous_dir / continuous_filename(START + 2000, "chrome"))
    assert meeting.artifacts()["diarization_stale"]
    assert not meeting.capabilities()["combine"]
    with pytest.raises(ValueError, match="Recording parts changed"):
        combine_meeting(meeting)
    diarize_meeting(meeting)
    assert meeting.capabilities()["combine"]


def test_added_microphone_clip_marks_transcription_stale(meeting, monkeypatch):
    first = wav(meeting.recordings_dir / segment_filename(START, "mic"))
    class Model:
        def transcribe(self, chunk):
            return TranscriptResult("mic", "speech", chunk.started_at, chunk.ended_at, .1)
    monkeypatch.setattr("avilistener.transcriber.build_transcriber", lambda config: Model())
    transcribe_meeting(meeting)
    assert not meeting.artifacts()["transcription_stale"]
    wav(meeting.recordings_dir / segment_filename(START + 1000, "mic"))
    assert meeting.artifacts()["transcription_stale"]
    transcribe_meeting(meeting)
    assert not meeting.artifacts()["transcription_stale"]
