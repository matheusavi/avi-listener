from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from avilistener.recorder import write_wav
from avilistener.transcriber import _is_ignored_text, build_transcriber


@dataclass(frozen=True)
class DiarizationTurn:
    start: float
    end: float
    speaker: str


DIARIZATION_ENGINES = ("nemo", "pyannote")


def run_meeting_diarize(
    audio_path: Path,
    config: dict,
    output_dir: Path,
    num_speakers: int | None,
    max_speakers: int,
    nemo_python: Path | None = None,
    speaker_names: dict[str, str] | None = None,
    engine: str = "nemo",
    hf_token: str | None = None,
) -> None:
    """Split one mixed recording into speakers and transcribe each of them.

    `config` is the transcription settings to use. They come from the meeting
    being diarized, which is where the dashboard keeps them.
    """
    if engine not in DIARIZATION_ENGINES:
        raise SystemExit(f"Unknown diarization engine {engine!r}; expected one of {DIARIZATION_ENGINES}.")
    output_dir.mkdir(parents=True, exist_ok=True)

    prepared = prepare_for_diarization(audio_path, output_dir)
    if engine == "pyannote":
        rttm_path = run_pyannote(prepared, output_dir, num_speakers, max_speakers, hf_token=hf_token)
    else:
        nemo_python = nemo_python or Path(".venv-nemo/Scripts/python.exe")
        if not nemo_python.exists():
            raise SystemExit(f"NeMo Python not found: {nemo_python}. Create .venv-nemo and install nemo_toolkit[asr].")
        rttm_path = run_nemo(prepared, output_dir, num_speakers, max_speakers, nemo_python)

    # Name the RTTM after the recording the user gave us, never the boosted
    # copy. `combine` reads the session start out of this filename, and the
    # boost is an implementation detail that must not leak into it.
    desired_rttm = output_dir / f"{audio_path.stem}.rttm"
    if rttm_path.resolve() != desired_rttm.resolve():
        shutil.copyfile(rttm_path, desired_rttm)
        rttm_path.unlink(missing_ok=True)
        rttm_path = desired_rttm
    if prepared != audio_path:
        prepared.unlink(missing_ok=True)  # a boosted copy of a long meeting is large
    turns = parse_rttm(rttm_path)
    from avilistener.timeline import load_timeline
    timeline = load_timeline(output_dir)
    if timeline:
        turns = [
            DiarizationTurn(max(turn.start, part["offset"]), min(turn.end, part["offset"] + part["duration"]), turn.speaker)
            for turn in turns for part in timeline["parts"]
            if min(turn.end, part["offset"] + part["duration"]) > max(turn.start, part["offset"])
        ]
        # Samples and labels use the same boundary-clipped turns.
        rttm_path.write_text("".join(
            f"SPEAKER {audio_path.stem} 1 {t.start:.6f} {t.end - t.start:.6f} <NA> <NA> {t.speaker} <NA> <NA>\n"
            for t in turns
        ), encoding="utf-8")
    write_turns(output_dir / "diarization.json", turns)

    # A previous run over this same audio already produced the words; changing
    # the engine or the speaker count only changes how they are split.
    words_path = output_dir / "words.json"
    write_speaker_transcript(
        audio_path, turns, output_dir, config, speaker_names,
        words_path=words_path if words_path.exists() else None,
    )


def write_speaker_transcript(
    audio_path: Path,
    turns: list[DiarizationTurn],
    output_dir: Path,
    config: dict,
    speaker_names: dict[str, str] | None = None,
    words_path: Path | None = None,
) -> None:
    """Transcribe with word timestamps and cut lines at speaker changes.

    Words are persisted to words.json: transcription is the expensive step, and
    keeping the words means better turns (a re-run with the right speaker
    count, or turns read from a video) can re-split without the GPU. Pass a
    previous run's words.json as `words_path` to do exactly that.
    """
    if words_path is not None and Path(words_path).exists():
        stored = json.loads(Path(words_path).read_text(encoding="utf-8"))
        all_words = [(float(w["start"]), float(w["end"]), str(w["word"])) for w in stored]
        fallback_path = Path(words_path).with_name("unaligned.json")
        fallback = json.loads(fallback_path.read_text(encoding="utf-8")) if fallback_path.exists() else []
        unaligned = [SpeakerLine(speaker_for_interval(turns, w["start"], w["end"]), w["text"], w["start"], w["end"]) for w in fallback]
        _emit_speaker_lines(all_words, unaligned, turns, output_dir, speaker_names or {})
        return

    transcriber = build_transcriber(config)
    speaker_names = speaker_names or {}
    segments, _info = transcriber.model.transcribe(
        str(audio_path),
        language=transcriber.language,
        beam_size=transcriber.options.beam_size,
        best_of=transcriber.options.best_of,
        patience=transcriber.options.patience,
        temperature=transcriber.options.temperature,
        condition_on_previous_text=transcriber.options.condition_on_previous_text,
        vad_filter=transcriber.options.vad_filter,
        vad_parameters=transcriber.options.vad_parameters,
        initial_prompt=transcriber.options.initial_prompt,
        hotwords=transcriber.options.hotwords,
        no_speech_threshold=transcriber.options.no_speech_threshold,
        log_prob_threshold=transcriber.options.log_prob_threshold,
        compression_ratio_threshold=transcriber.options.compression_ratio_threshold,
        repetition_penalty=transcriber.options.repetition_penalty,
        no_repeat_ngram_size=transcriber.options.no_repeat_ngram_size,
        hallucination_silence_threshold=transcriber.options.hallucination_silence_threshold,
        # Needed to cut lines at speaker changes rather than at whatever
        # boundaries Whisper happened to choose.
        word_timestamps=True,
    )

    # Collect every word first, then split once. Splitting per segment would
    # leave a speaker's line broken wherever Whisper happened to end a segment,
    # since runs either side of that boundary could never be joined.
    all_words: list[tuple[float, float, str]] = []
    unaligned: list[SpeakerLine] = []
    for segment in segments:
        text = segment.text.strip()
        if not text:
            continue
        if _is_ignored_text(text, transcriber.options.ignored_phrases):
            continue

        words = [(word.start, word.end, word.word) for word in (segment.words or []) if word.start is not None]
        if words:
            all_words.extend(words)
        else:
            # No word timings: fall back to labelling the whole segment.
            speaker = speaker_for_interval(turns, segment.start, segment.end)
            unaligned.append(SpeakerLine(speaker, text, segment.start, segment.end))

    _emit_speaker_lines(all_words, unaligned, turns, output_dir, speaker_names)


def _emit_speaker_lines(
    all_words: list[tuple[float, float, str]],
    unaligned: list[SpeakerLine],
    turns: list[DiarizationTurn],
    output_dir: Path,
    speaker_names: dict[str, str],
) -> None:
    from dataclasses import asdict
    (output_dir / "unaligned.json").write_text(json.dumps([asdict(line) for line in unaligned], ensure_ascii=False), encoding="utf-8")
    (output_dir / "words.json").write_text(
        json.dumps([{"start": s, "end": e, "word": w} for s, e, w in all_words], ensure_ascii=False),
        encoding="utf-8",
    )
    from avilistener.timeline import load_timeline, part_at, wall_time

    timeline = load_timeline(output_dir)
    parts = timeline["parts"] if timeline else []
    if parts:
        # Never join speech across a restart, even if the same person is speaking.
        groups = [[] for _ in parts]
        for start, end, word in all_words:
            index = part_at(parts, (start + end) / 2)
            part = parts[index]
            groups[index].append((max(start, part["offset"]), min(end, part["offset"] + part["duration"]), word))
        speaker_lines = [line for words in groups for line in split_words_by_speaker(words, turns)] + unaligned
    else:
        speaker_lines = split_words_by_speaker(all_words, turns) + unaligned
    speaker_lines.sort(key=lambda item: item.start)

    lines: list[str] = []
    events: list[dict] = []
    for speaker_line in speaker_lines:
        display_speaker = speaker_names.get(speaker_line.speaker, speaker_line.speaker)
        lines.append(
            f"[{format_seconds(speaker_line.start)} - {format_seconds(speaker_line.end)}] "
            f"{display_speaker:<12} | {speaker_line.text}"
        )
        events.append(
            {
                "start": speaker_line.start,
                "end": speaker_line.end,
                "speaker": display_speaker,
                "speaker_label": speaker_line.speaker,
                "text": speaker_line.text,
            }
        )
        if parts:
            index = part_at(parts, (speaker_line.start + speaker_line.end) / 2)
            part = parts[index]
            events[-1].update({
                "part": index,
                "started_at": wall_time(part, speaker_line.start),
                "ended_at": wall_time(part, speaker_line.end),
            })

    (output_dir / "combined.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output_dir / "events.json").write_text(json.dumps(events, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote diarized transcript to {output_dir / 'combined.md'}")


@dataclass(frozen=True)
class SpeakerLine:
    speaker: str
    text: str
    start: float
    end: float


def split_words_by_speaker(
    words: list[tuple[float, float, str]],
    turns: list[DiarizationTurn],
    min_line_seconds: float = 0.6,
) -> list[SpeakerLine]:
    """Cut a transcript into lines wherever the speaker changes.

    Whisper emits segments of its own choosing, routinely 10-15 seconds long,
    and a single segment often spans several diarization turns. Labelling whole
    segments therefore throws away every speaker change inside one, and a
    participant whose turns are all short can vanish from the transcript
    entirely. Assigning each word separately keeps them.

    Very short runs are folded back into the previous line: diarization
    boundaries are approximate, and a stray word landing across one would
    otherwise flip speaker for a fraction of a second.
    """
    lines: list[SpeakerLine] = []
    for start, end, text in words:
        text = text.strip()
        if not text:
            continue
        speaker = speaker_for_interval(turns, start, end)
        if lines and lines[-1].speaker == speaker:
            previous = lines[-1]
            lines[-1] = SpeakerLine(speaker, f"{previous.text} {text}", previous.start, end)
        else:
            lines.append(SpeakerLine(speaker, text, start, end))

    if len(lines) < 2:
        return lines

    smoothed: list[SpeakerLine] = []
    for index, line in enumerate(lines):
        previous = smoothed[-1] if smoothed else None
        following = lines[index + 1] if index + 1 < len(lines) else None

        word_count = len(line.text.split())
        brief = word_count <= 1 or ((line.end - line.start) < min_line_seconds and word_count <= 2)
        # Only absorb a brief run when the same speaker holds the floor either
        # side of it: that pattern is a boundary artefact, not a turn. Filler
        # like "e" or "um" often carries a timestamp spanning the pause around
        # it, so duration alone would let it split a sentence in half. A brief
        # run between two *different* speakers is a real interjection and stays.
        is_flicker = (
            brief
            and previous is not None
            and following is not None
            and previous.speaker == following.speaker
            and line.speaker != previous.speaker
        )

        if previous is not None and (is_flicker or line.speaker == previous.speaker):
            smoothed[-1] = SpeakerLine(previous.speaker, f"{previous.text} {line.text}", previous.start, line.end)
        else:
            smoothed.append(line)
    return smoothed


def diarization_gain(audio: np.ndarray, target: float = 0.7, max_gain: float = 60.0) -> float:
    """How much to amplify audio so NeMo's VAD can hear it.

    Unlike Whisper, `vad_multilingual_marblenet` judges absolute level, so a
    quiet recording is read as silence and produces no speaker turns at all.
    Everything Whisper transcribes outside those turns then falls back to
    SPEAKER_UNKNOWN.

    A high percentile stands in for the peak so that a single click, which
    dropped samples readily produce, cannot suppress the gain for the whole
    recording. Audio that is already loud is left alone rather than attenuated.
    """
    if audio.size == 0:
        return 1.0
    loud = float(np.percentile(np.abs(audio), 99.9))
    if loud <= 1e-6:
        return 1.0  # silence: amplifying it would only raise the noise
    return float(min(max(target / loud, 1.0), max_gain))


def prepare_for_diarization(audio_path: Path, work_dir: Path) -> Path:
    """Return audio loud enough for NeMo, leaving the original untouched.

    Only diarization gets the boosted copy. Whisper normalises internally and
    already transcribes quiet audio correctly, and amplifying its input would
    raise the noise floor into the range where it hallucinates over silence.
    """
    try:
        with wave.open(str(audio_path), "rb") as handle:
            channels = handle.getnchannels()
            sample_width = handle.getsampwidth()
            sample_rate = handle.getframerate()
            frames = handle.readframes(handle.getnframes())
    except (wave.Error, EOFError, OSError):
        return audio_path  # not a plain WAV; let NeMo's own loader handle it
    if sample_width != 2 or not frames:
        return audio_path

    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)

    gain = diarization_gain(audio)
    if gain <= 1.01:
        return audio_path

    boosted_path = work_dir / f"{audio_path.stem}.boosted.wav"
    write_wav(boosted_path, np.clip(audio * gain, -1.0, 1.0), sample_rate)
    print(f"Boosted diarization audio {gain:.1f}x (+{20 * math.log10(gain):.1f} dB) so VAD can detect speech")
    return boosted_path


def run_pyannote(
    audio_path: Path,
    output_dir: Path,
    num_speakers: int | None,
    max_speakers: int,
    pyannote_python: Path | None = None,
    hf_token: str | None = None,
) -> Path:
    pyannote_python = pyannote_python or Path(".venv-pyannote/Scripts/python.exe")
    if not pyannote_python.exists():
        raise SystemExit(
            f"pyannote Python not found: {pyannote_python}. "
            "Create .venv-pyannote and install torch (cu128) plus pyannote.audio."
        )
    script = Path(__file__).resolve().parent.parent / "scripts" / "pyannote_diarize.py"
    cmd = [
        str(pyannote_python),
        str(script),
        "--audio",
        str(audio_path),
        "--out",
        str(output_dir),
        "--max-speakers",
        str(max_speakers),
    ]
    if num_speakers is not None:
        cmd.extend(["--num-speakers", str(num_speakers)])
    # A stored token travels in the environment, never on the command line,
    # where it would be visible to anything that can list processes.
    env = dict(os.environ)
    if hf_token:
        env["HF_TOKEN"] = hf_token
    completed = subprocess.run(cmd, text=True, capture_output=True, env=env)
    if completed.returncode != 0:
        sys.stderr.write(completed.stdout)
        sys.stderr.write(completed.stderr)
        raise SystemExit(completed.returncode)
    last_line = completed.stdout.strip().splitlines()[-1]
    return Path(last_line)


def run_nemo(
    audio_path: Path,
    output_dir: Path,
    num_speakers: int | None,
    max_speakers: int,
    nemo_python: Path,
) -> Path:
    script = Path(__file__).resolve().parent.parent / "scripts" / "nemo_diarize.py"
    cmd = [
        str(nemo_python),
        str(script),
        "--audio",
        str(audio_path),
        "--out",
        str(output_dir),
        "--max-speakers",
        str(max_speakers),
    ]
    if num_speakers is not None:
        cmd.extend(["--num-speakers", str(num_speakers)])
    completed = subprocess.run(cmd, text=True, capture_output=True)
    if completed.returncode != 0:
        sys.stderr.write(completed.stdout)
        sys.stderr.write(completed.stderr)
        raise SystemExit(completed.returncode)
    last_line = completed.stdout.strip().splitlines()[-1]
    return Path(last_line)


def parse_rttm(path: Path) -> list[DiarizationTurn]:
    turns: list[DiarizationTurn] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 8 or parts[0] != "SPEAKER":
            continue
        start = float(parts[3])
        duration = float(parts[4])
        speaker = parts[7]
        turns.append(DiarizationTurn(start=start, end=start + duration, speaker=speaker))
    return sorted(turns, key=lambda item: (item.start, item.end))


def speaker_for_interval(turns: list[DiarizationTurn], start: float, end: float) -> str:
    midpoint = (start + end) / 2
    best = None
    best_overlap = 0.0
    for turn in turns:
        if turn.start <= midpoint <= turn.end:
            return turn.speaker
        overlap = max(0.0, min(end, turn.end) - max(start, turn.start))
        if overlap > best_overlap:
            best = turn
            best_overlap = overlap
    return best.speaker if best is not None else "SPEAKER_UNKNOWN"


def write_turns(path: Path, turns: list[DiarizationTurn]) -> None:
    payload = [{"start": t.start, "end": t.end, "speaker": t.speaker} for t in turns]
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def format_seconds(value: float) -> str:
    minutes, seconds = divmod(value, 60)
    hours, minutes = divmod(int(minutes), 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:05.2f}"
