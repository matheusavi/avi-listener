"""Meeting processing that publishes complete results and retains prior outputs."""
from __future__ import annotations

import hashlib
import json
import shutil
import uuid
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from avilistener.server.workspace import Meeting
from avilistener.timeline import build_timeline, inspect_parts, load_timeline, sha256


def transcription_config(meeting: Meeting) -> dict:
    config = meeting.config
    return {
        "model_size": config.get("model_size", "large-v3"),
        "device": config.get("device", "cuda"),
        "compute_type": config.get("compute_type", "float16"),
        "language": config.get("language") or None,
        "transcription": config.get("transcription", {}) or {},
    }


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def staging(meeting: Meeting, kind: str) -> Path:
    path = meeting.path / ".processing" / f"{kind}-{uuid.uuid4().hex}"
    path.mkdir(parents=True)
    return path


def publish(stage: Path, destination: Path) -> None:
    """Preserve the last successful output; never target recordings."""
    if destination.name not in {"transcripts", "diarized", "merged", "speaker-samples"}:
        raise ValueError(f"Not a derived output directory: {destination}")
    previous = None
    if destination.exists():
        history = destination.parent / "artifact-history"
        history.mkdir(exist_ok=True)
        previous = history / f"{datetime.now():%Y%m%d-%H%M%S}-{destination.name}-{uuid.uuid4().hex[:8]}"
        destination.rename(previous)
    try:
        stage.rename(destination)
    except BaseException:
        if previous is not None:
            previous.rename(destination)
        raise


def transcribe_meeting(meeting: Meeting, log=print) -> None:
    from avilistener.file_transcriber import collect_wav_clips, source_name_from_discord_wav, wav_to_audio_chunk
    from avilistener.transcriber import TranscriptResult, build_transcriber
    from avilistener.writer import TranscriptWriter

    config = transcription_config(meeting)
    paths = collect_wav_clips(meeting.recordings_dir)
    if not paths:
        raise ValueError("No speech clips found. Continuous recordings can be split by speaker directly.")
    stage = staging(meeting, "transcripts")
    writer = TranscriptWriter(stage)
    writer.events_path.touch()
    writer.combined_path.touch()
    cache_dir = meeting.path / ".processing" / "clip-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    transcriber = None
    count = 0
    inputs = []
    for index, path in enumerate(paths):
        digest = sha256(path)
        inputs.append({"filename": path.name, "bytes": path.stat().st_size, "sha256": digest})
        cache = cache_dir / f"{fingerprint([config, path.name, digest])}.json"
        if cache.exists():
            stored = json.loads(cache.read_text(encoding="utf-8"))
            result = TranscriptResult(**stored) if stored else None
        else:
            if transcriber is None:
                log("Loading transcription model...")
                transcriber = build_transcriber(config)
            chunk = wav_to_audio_chunk(source_name_from_discord_wav(path), path)
            result = transcriber.transcribe(chunk)
            if sha256(path) != digest:
                raise ValueError(f"Recording changed while transcribing: {path.name}")
            temporary = cache.with_suffix(".tmp")
            temporary.write_text(json.dumps(asdict(result) if result else None, ensure_ascii=False), encoding="utf-8")
            temporary.replace(cache)
        if result:
            writer.write(result)
            count += 1
        if (index + 1) % 25 == 0 or index + 1 == len(paths):
            log(f"Transcribed {index + 1}/{len(paths)} clips ({count} with speech)")
    (stage / "inputs.json").write_text(json.dumps(inputs, indent=2), encoding="utf-8")
    publish(stage, meeting.transcripts_dir)


def diarize_meeting(meeting: Meeting, num_speakers=None, max_speakers=6, engine=None, log=print) -> None:
    from avilistener.meeting import run_meeting_diarize

    paths = meeting.shared_parts()
    parts = inspect_parts(paths)
    config = transcription_config(meeting)
    key = fingerprint({"parts": parts, "config": config, "timeline_version": 1})
    stage = staging(meeting, "diarized")
    log(f"Preparing {len(parts)} recording part(s); originals remain in place")
    audio = build_timeline(paths, stage, parts)
    cache_key = meeting.diarized_dir / "cache-key.json"
    if cache_key.exists() and json.loads(cache_key.read_text()) == key:
        for name in ("words.json", "unaligned.json"):
            previous = meeting.diarized_dir / name
            if previous.exists():
                shutil.copy2(previous, stage / name)
        log("Reusing words for identical audio parts and transcription settings")
    engine = engine or str(meeting.config.get("diarization_engine") or "pyannote")
    log(f"Splitting {sum(p['duration'] for p in parts):.1f}s across all parts with {engine}")
    run_meeting_diarize(
        audio_path=audio, config=config, output_dir=stage,
        num_speakers=num_speakers, max_speakers=max_speakers,
        engine=engine, hf_token=str(meeting.config.get("hf_token") or "").strip() or None,
    )
    (stage / "cache-key.json").write_text(json.dumps(key), encoding="utf-8")
    (stage / "previous-speaker-names.json").write_text(
        json.dumps(meeting.config.get("speaker_names") or {}, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    # Prior names are not carried across a fresh clustering: label numbers may change.
    samples = stage / "speaker-samples"
    samples.mkdir()
    from avilistener.meeting import parse_rttm
    from avilistener.speakers import extract_wav_slice, pick_sample_turn, speaker_labels
    turns = parse_rttm(stage / "session-audio.rttm")
    for label in speaker_labels(turns):
        span = pick_sample_turn(turns, label)
        if span:
            extract_wav_slice(audio, samples / f"{label}.wav", *span)
    # Check the inputs again before publishing a result for this exact recording set.
    if meeting.shared_parts() != paths or any(sha256(p) != part["sha256"] for p, part in zip(paths, parts)):
        raise ValueError("Recording inputs changed; stop recording and try again")
    publish(stage, meeting.diarized_dir)
    publish(meeting.diarized_dir / "speaker-samples", meeting.path / "speaker-samples")
    meeting.update_config({"speaker_names": {}})
    log(f"Diarization finished: {len(parts)} parts, {len(speaker_labels(turns))} speakers")


def combine_meeting(meeting: Meeting, log=print) -> None:
    from avilistener.combine import find_diarized_source, load_diarized_events, load_transcript_events, merge_results
    from avilistener.writer import TranscriptWriter

    artifacts = meeting.artifacts()
    if artifacts["diarization_stale"] or artifacts["transcription_stale"]:
        raise ValueError("Recording parts changed. Transcribe and split again before merging.")
    source = find_diarized_source(meeting.diarized_dir)
    timeline = load_timeline(meeting.diarized_dir)
    diarized = load_diarized_events(meeting.diarized_dir, source.started_at, names=meeting.config.get("speaker_names") or {})
    coverage = timeline["parts"] if timeline else None
    others = load_transcript_events(
        meeting.transcripts_dir / "events.jsonl", exclude_source=source.source_name,
        rename={"mic": meeting.config.get("me") or "host"}, coverage=coverage,
    )
    stage = staging(meeting, "merged")
    writer = TranscriptWriter(stage)
    writer.events_path.touch()
    writer.combined_path.touch()
    results = merge_results(diarized, others)
    for result in results:
        writer.write(result)
    if timeline:
        (stage / "timeline.json").write_text(json.dumps(timeline, indent=2), encoding="utf-8")
    publish(stage, meeting.merged_dir)
    log(f"Merged {len(results)} lines on the original wall-clock timeline")
