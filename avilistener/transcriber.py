from __future__ import annotations

import os
import re
from pathlib import Path
from dataclasses import dataclass
from typing import Any

from avilistener.audio import AudioChunk


@dataclass(frozen=True)
class TranscriptResult:
    source: str
    text: str
    started_at: float
    ended_at: float
    rms: float


@dataclass(frozen=True)
class TranscriptionOptions:
    beam_size: int = 5
    best_of: int = 5
    patience: float = 1.0
    temperature: float | list[float] = 0.0
    condition_on_previous_text: bool = False
    vad_filter: bool = True
    vad_parameters: dict[str, Any] | None = None
    initial_prompt: str | None = None
    hotwords: str | None = None
    no_speech_threshold: float | None = 0.6
    log_prob_threshold: float | None = -1.0
    compression_ratio_threshold: float | None = 2.4
    repetition_penalty: float = 1.0
    no_repeat_ngram_size: int = 0
    hallucination_silence_threshold: float | None = None
    ignored_phrases: tuple[str, ...] = ()


class Transcriber:
    def __init__(
        self,
        model_size: str,
        device: str,
        compute_type: str,
        language: str | None,
        options: TranscriptionOptions | None = None,
    ) -> None:
        if device == "cuda":
            _add_cuda_dll_directories()
        from faster_whisper import WhisperModel

        self.language = language or None
        self.options = options or TranscriptionOptions()
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe(self, chunk: AudioChunk) -> TranscriptResult | None:
        segments, _info = self.model.transcribe(
            chunk.audio,
            language=self.language,
            beam_size=self.options.beam_size,
            best_of=self.options.best_of,
            patience=self.options.patience,
            temperature=self.options.temperature,
            condition_on_previous_text=self.options.condition_on_previous_text,
            vad_filter=self.options.vad_filter,
            vad_parameters=self.options.vad_parameters,
            initial_prompt=self.options.initial_prompt,
            hotwords=self.options.hotwords,
            no_speech_threshold=self.options.no_speech_threshold,
            log_prob_threshold=self.options.log_prob_threshold,
            compression_ratio_threshold=self.options.compression_ratio_threshold,
            repetition_penalty=self.options.repetition_penalty,
            no_repeat_ngram_size=self.options.no_repeat_ngram_size,
            hallucination_silence_threshold=self.options.hallucination_silence_threshold,
        )
        text = " ".join(segment.text.strip() for segment in segments).strip()
        if not text:
            return None
        if _is_ignored_text(text, self.options.ignored_phrases):
            return None
        return TranscriptResult(
            source=chunk.source,
            text=text,
            started_at=chunk.started_at,
            ended_at=chunk.ended_at,
            rms=chunk.rms,
        )


def _is_ignored_text(text: str, ignored_phrases: tuple[str, ...]) -> bool:
    normalized = " ".join(text.casefold().split())
    if any(phrase.casefold() in normalized for phrase in ignored_phrases):
        return True
    return _looks_like_prompt_leak(normalized) or _looks_like_boilerplate_hallucination(normalized) or _looks_repetitive(normalized)


def _looks_like_prompt_leak(normalized: str) -> bool:
    prompt_fragments = (
        "a conversa pode mencionar",
        "conversa casual em discord",
        "preserve nomes de usuarios",
        "transcreva apenas o que foi dito",
        "nao invente frases para silencio",
        "não invente frases para silêncio",
    )
    return any(fragment in normalized for fragment in prompt_fragments)


def _looks_like_boilerplate_hallucination(normalized: str) -> bool:
    boilerplate_patterns = (
        r"\binscreva-se\b.*\bcanal\b",
        r"\bse inscreva\b.*\bcanal\b",
        r"\bative\b.*\bsininho\b",
        r"\breceber notificações\b",
        r"\bobrigad[oa] por assistir\b",
        r"\bthank you for watching\b",
        r"\bum abraço e até a próxima\b",
    )
    return any(re.search(pattern, normalized) for pattern in boilerplate_patterns)


def _looks_repetitive(normalized: str) -> bool:
    words = re.findall(r"\w+", normalized, flags=re.UNICODE)
    if len(words) < 18:
        return False

    unique_ratio = len(set(words)) / len(words)
    if unique_ratio < 0.35:
        return True

    stems = [word[:5] for word in words if len(word) >= 5]
    if not stems:
        return False
    most_common_stem_count = max(stems.count(stem) for stem in set(stems))
    return most_common_stem_count >= 10 and most_common_stem_count / len(words) >= 0.3


def _add_cuda_dll_directories() -> None:
    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return

    candidates: list[Path] = []
    for env_name in ("CUDA_PATH", "CUDA_HOME"):
        env_path = os.environ.get(env_name)
        if env_path:
            candidates.append(Path(env_path) / "bin")

    cuda_root = Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA")
    if cuda_root.exists():
        candidates.extend(sorted((path / "bin" for path in cuda_root.glob("v*")), reverse=True))

    for path in candidates:
        if path.exists():
            os.environ["PATH"] = f"{path}{os.pathsep}{os.environ.get('PATH', '')}"
            os.add_dll_directory(str(path))


def build_transcriber(config: dict) -> Transcriber:
    raw_options = config.get("transcription", {}) or {}
    return Transcriber(
        model_size=str(config.get("model_size", "large-v3")),
        device=str(config.get("device", "cuda")),
        compute_type=str(config.get("compute_type", "float16")),
        language=config.get("language"),
        options=TranscriptionOptions(
            beam_size=int(raw_options.get("beam_size", 5)),
            best_of=int(raw_options.get("best_of", 5)),
            patience=float(raw_options.get("patience", 1.0)),
            temperature=_temperature(raw_options.get("temperature", 0.0)),
            condition_on_previous_text=bool(raw_options.get("condition_on_previous_text", False)),
            vad_filter=bool(raw_options.get("vad_filter", True)),
            vad_parameters=raw_options.get("vad_parameters"),
            initial_prompt=raw_options.get("initial_prompt") or None,
            hotwords=_hotwords(raw_options.get("hotwords")),
            no_speech_threshold=_optional_float(raw_options.get("no_speech_threshold", 0.6)),
            log_prob_threshold=_optional_float(raw_options.get("log_prob_threshold", -1.0)),
            compression_ratio_threshold=_optional_float(raw_options.get("compression_ratio_threshold", 2.4)),
            repetition_penalty=float(raw_options.get("repetition_penalty", 1.0)),
            no_repeat_ngram_size=int(raw_options.get("no_repeat_ngram_size", 0)),
            hallucination_silence_threshold=_optional_float(raw_options.get("hallucination_silence_threshold")),
            ignored_phrases=tuple(str(item) for item in raw_options.get("ignored_phrases", []) or []),
        ),
    )


def _temperature(value) -> float | list[float]:
    if isinstance(value, list):
        return [float(item) for item in value]
    return float(value)


def _hotwords(value) -> str | None:
    if not value:
        return None
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    return str(value)


def _optional_float(value) -> float | None:
    if value in (None, ""):
        return None
    return float(value)
