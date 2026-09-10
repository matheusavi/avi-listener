from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import soundcard as sc


@dataclass(frozen=True)
class SourceConfig:
    name: str
    kind: str
    enabled: bool = True
    device: str | None = None


@dataclass(frozen=True)
class AudioChunk:
    source: str
    audio: np.ndarray
    sample_rate: int
    started_at: float
    ended_at: float
    rms: float


def list_audio_devices() -> str:
    lines: list[str] = []
    lines.append("Speakers / loopback-capable outputs:")
    for speaker in sc.all_speakers():
        default_mark = " (default)" if speaker.name == sc.default_speaker().name else ""
        lines.append(f"  - {speaker.name}{default_mark}")

    lines.append("")
    lines.append("Microphones / inputs:")
    for mic in sc.all_microphones(include_loopback=True):
        default_mark = " (default)" if mic.name == sc.default_microphone().name else ""
        lines.append(f"  - {mic.name}{default_mark}")
    return "\n".join(lines)


def list_devices_structured() -> dict:
    """Devices as data, for a picker rather than a printed table.

    Loopback entries are excluded from the microphone list: they are outputs
    being captured, and offering them as a microphone invites recording the
    call back into itself.
    """
    speakers = []
    default_speaker = None
    try:
        default_speaker = sc.default_speaker().name
    except Exception:
        pass
    for speaker in sc.all_speakers():
        speakers.append({"name": speaker.name, "default": speaker.name == default_speaker})

    microphones = []
    default_mic = None
    try:
        default_mic = sc.default_microphone().name
    except Exception:
        pass
    speaker_names = {item["name"] for item in speakers}
    for mic in sc.all_microphones(include_loopback=False):
        if mic.name in speaker_names:
            continue
        microphones.append({"name": mic.name, "default": mic.name == default_mic})

    return {"microphones": microphones, "speakers": speakers}


def resolve_recorder_source(source: SourceConfig):
    if source.kind == "microphone":
        if source.device:
            return _find_microphone(source.device, include_loopback=False)
        return sc.default_microphone()

    if source.kind == "loopback":
        speaker = _find_speaker(source.device) if source.device else sc.default_speaker()
        return sc.get_microphone(speaker.name, include_loopback=True)

    raise ValueError(f"Unsupported source kind for {source.name!r}: {source.kind!r}")


def _find_speaker(name: str | None):
    if not name:
        return sc.default_speaker()
    needle = name.lower()
    speakers = sc.all_speakers()
    for speaker in speakers:
        if speaker.name.lower() == needle:
            return speaker
    for speaker in speakers:
        if needle in speaker.name.lower():
            return speaker
    available = ", ".join(s.name for s in speakers)
    raise ValueError(f"Could not find speaker containing {name!r}. Available: {available}")


def _find_microphone(name: str, include_loopback: bool):
    needle = name.lower()
    microphones = sc.all_microphones(include_loopback=include_loopback)
    for mic in microphones:
        if mic.name.lower() == needle:
            return mic
    for mic in microphones:
        if needle in mic.name.lower():
            return mic
    available = ", ".join(m.name for m in microphones)
    raise ValueError(f"Could not find microphone containing {name!r}. Available: {available}")


def enabled_sources(raw_sources: dict) -> Iterable[SourceConfig]:
    for name, raw in raw_sources.items():
        source = SourceConfig(
            name=name,
            kind=str(raw.get("kind", "")).strip(),
            enabled=bool(raw.get("enabled", True)),
            device=raw.get("device") or None,
        )
        if source.enabled:
            yield source

