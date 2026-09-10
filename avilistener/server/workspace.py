"""Projects and meetings as folders on disk.

A project groups related meetings and holds the defaults they inherit; a
meeting owns one recording session and everything derived from it. Keeping
each meeting self-contained is what makes it safe to run several with
different settings: nothing is shared, so a second session cannot pick up the
first one's audio or diarization.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Kept out of every API response. The token is a credential: it is stored so
# recording can start without retyping it, but sending it back to the browser
# would put it in logs, caches and devtools for no benefit.
SECRET_KEYS = {"discord_token", "hf_token"}


def public_config(config: dict) -> dict:
    """Config safe to hand to the interface, with secrets replaced by a flag."""
    visible = {key: value for key, value in config.items() if key not in SECRET_KEYS}
    for key in SECRET_KEYS:
        visible[f"{key}_set"] = bool(str(config.get(key) or "").strip())
    return visible


DEFAULT_PROJECT_CONFIG: dict[str, Any] = {
    "model_size": "large-v3",
    "device": "cuda",
    "compute_type": "float16",
    "language": "pt",
    "sources": ["mic", "system"],
    "me": "host",
    "num_speakers": None,
    # pyannote/speaker-diarization-community-1 is the default: measured on a
    # 2h51m 5-voice session it found all five speakers unaided where NeMo
    # found three with the count forced, and ran in 2 minutes instead of 18.
    # It needs .venv-pyannote and a Hugging Face token; starting a split
    # without one fails with instructions, and "nemo" remains selectable.
    "diarization_engine": "pyannote",
    "hf_token": "",
    "sample_rate": 16000,
    "devices": {},
    "discord_token": "",
    "discord_channel_id": "",
    # Tuned rather than left to faster-whisper's defaults, which produce
    # noticeably worse transcripts here: no hallucination filtering, no
    # hotwords for names, and looser thresholds.
    "transcription": {
        "beam_size": 8,
        "best_of": 8,
        "patience": 1.2,
        "temperature": 0.0,
        "condition_on_previous_text": False,
        "vad_filter": True,
        "vad_parameters": {"min_silence_duration_ms": 350, "speech_pad_ms": 250},
        "no_speech_threshold": 0.75,
        "log_prob_threshold": -0.8,
        "compression_ratio_threshold": 2.2,
        "repetition_penalty": 1.05,
        "no_repeat_ngram_size": 3,
        "hallucination_silence_threshold": 0.5,
        # Names and jargon Whisper otherwise spells inconsistently.
        "hotwords": [],
        # Whisper invents these over silence; they are never really said.
        "ignored_phrases": [
            "thank you for watching",
            "obrigado por assistir",
            "se inscreva no canal",
            "inscreva-se no canal",
            "ative o sininho",
            "receber notificações de novos vídeos",
            "legendas por",
            "legendas pela comunidade",
        ],
        "initial_prompt": "",
    },
}


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip()).strip("-.")
    return (slug[:60] or "untitled").lower()


def _read_json(path: Path, fallback: dict) -> dict:
    if not path.exists():
        return dict(fallback)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return dict(fallback)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


@dataclass
class Meeting:
    project: "Project"
    slug: str
    path: Path

    @property
    def meta_path(self) -> Path:
        return self.path / "meeting.json"

    @property
    def meta(self) -> dict:
        return _read_json(self.meta_path, {"name": self.slug, "config": {}, "created_at": 0})

    @property
    def name(self) -> str:
        return str(self.meta.get("name") or self.slug)

    @property
    def config(self) -> dict:
        """Project defaults with this meeting's overrides applied."""
        merged = dict(self.project.config)
        merged.update(self.meta.get("config") or {})
        return merged

    def update_config(self, changes: dict) -> None:
        meta = self.meta
        config = dict(meta.get("config") or {})
        config.update({key: value for key, value in changes.items() if value is not None})
        meta["config"] = config
        _write_json(self.meta_path, meta)

    # Directory layout, one place so nothing has to guess at paths.
    @property
    def recordings_dir(self) -> Path:
        return self.path / "recordings"

    @property
    def continuous_dir(self) -> Path:
        return self.recordings_dir / "continuous"

    @property
    def transcripts_dir(self) -> Path:
        return self.path / "transcripts"

    @property
    def diarized_dir(self) -> Path:
        return self.path / "diarized"

    @property
    def merged_dir(self) -> Path:
        return self.path / "merged"

    @property
    def logs_dir(self) -> Path:
        return self.path / "logs"

    def continuous_for(self, source: str) -> Path | None:
        matches = self.continuous_parts(source)
        return matches[-1] if matches else None

    def continuous_parts(self, source: str) -> list[Path]:
        return sorted(self.continuous_dir.glob(f"*-{source}-continuous.wav"))

    def shared_parts(self) -> list[Path]:
        return sorted(path for source in ("system", "chrome", "discord") for path in self.continuous_parts(source))

    def speaker_audio(self) -> Path | None:
        """Speaker turns refer to the compact working audio for multipart runs."""
        timeline = self.diarized_dir / "timeline.json"
        if timeline.exists():
            return self.diarized_dir / "session-audio.wav"
        # Legacy output must use the recording it actually describes, even if
        # another recording has since been added to this meeting.
        matches = list(self.diarized_dir.glob("*-continuous.rttm"))
        if len(matches) == 1:
            path = self.continuous_dir / matches[0].with_suffix(".wav").name
            if path.exists():
                return path
        return self.loopback_continuous()

    def loopback_continuous(self) -> Path | None:
        """The shared recording worth diarizing into individual speakers."""
        for source in ("system", "chrome", "discord"):
            found = self.continuous_for(source)
            if found is not None:
                return found
        return None

    def segment_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for wav in self.recordings_dir.glob("*.wav"):
            match = re.match(r"^\d{4}-\d{2}-\d{2}T[\d-]+Z-(.+)-\d+$", wav.stem)
            if match:
                counts[match.group(1)] = counts.get(match.group(1), 0) + 1
        for wav in (self.recordings_dir / "processed").glob("*.wav"):
            match = re.match(r"^\d{4}-\d{2}-\d{2}T[\d-]+Z-(.+)-\d+$", wav.stem)
            if match:
                counts[match.group(1)] = counts.get(match.group(1), 0) + 1
        return counts

    def artifacts(self) -> dict:
        segments = self.segment_counts()
        continuous = {
            source: (self.continuous_for(source) is not None)
            for source in ("mic", "system", "chrome", "discord")
        }
        shared = self.shared_parts()
        timeline_path = self.diarized_dir / "timeline.json"
        has_diarized = (self.diarized_dir / "events.json").exists()
        diarization_stale = False
        if has_diarized:
            if timeline_path.exists():
                timeline = _read_json(timeline_path, {"parts": []})
                expected = sorted((part["filename"], part["bytes"]) for part in timeline["parts"])
                diarization_stale = expected != sorted((path.name, path.stat().st_size) for path in shared)
            else:
                # Legacy output represents one recording only.
                diarization_stale = len(shared) > 1
        transcription_stale = False
        inputs_path = self.transcripts_dir / "inputs.json"
        if inputs_path.exists():
            inputs = json.loads(inputs_path.read_text(encoding="utf-8"))
            current = {path.name: path.stat().st_size for directory in (self.recordings_dir, self.recordings_dir / "processed")
                       for path in directory.glob("*.wav")}
            expected = {item["filename"]: item.get("bytes", current.get(item["filename"])) for item in inputs}
            transcription_stale = current != expected
        merged_path = self.merged_dir / "events.jsonl"
        merged_stale = (diarization_stale or transcription_stale or any(
            path.exists() and merged_path.exists() and path.stat().st_mtime_ns > merged_path.stat().st_mtime_ns
            for path in (self.transcripts_dir / "events.jsonl", self.diarized_dir / "events.json")
        ))
        return {
            "segments": segments,
            "continuous": continuous,
            "continuous_parts": {source: len(self.continuous_parts(source)) for source in continuous},
            "has_recording": bool(segments) or any(continuous.values()),
            "has_transcripts": (self.transcripts_dir / "events.jsonl").exists(),
            "has_diarized": has_diarized,
            "diarization_stale": diarization_stale,
            "transcription_stale": transcription_stale,
            "merged_stale": merged_stale,
            "has_merged": (self.merged_dir / "combined.txt").exists(),
            "loopback": self.loopback_continuous().name if self.loopback_continuous() else None,
        }

    def capabilities(self) -> dict:
        """What the user may do next.

        The UI greys out everything else, so an action can never be offered
        before the thing it consumes exists.
        """
        art = self.artifacts()
        return {
            "transcribe": art["has_recording"],
            # Diarization needs the loopback: it is the only stream with
            # several people mixed together.
            "diarize": art["loopback"] is not None,
            "combine": art["has_diarized"] and art["has_transcripts"] and not art["diarization_stale"] and not art["transcription_stale"],
        }

    def to_json(self) -> dict:
        return {
            "slug": self.slug,
            "name": self.name,
            "project": self.project.slug,
            "created_at": self.meta.get("created_at", 0),
            "config": public_config(self.config),
            "artifacts": self.artifacts(),
            "can": self.capabilities(),
        }


@dataclass
class Project:
    root: Path
    slug: str
    path: Path

    @property
    def meta_path(self) -> Path:
        return self.path / "project.json"

    @property
    def meta(self) -> dict:
        return _read_json(self.meta_path, {"name": self.slug, "config": dict(DEFAULT_PROJECT_CONFIG)})

    @property
    def name(self) -> str:
        return str(self.meta.get("name") or self.slug)

    @property
    def config(self) -> dict:
        merged = dict(DEFAULT_PROJECT_CONFIG)
        merged.update(self.meta.get("config") or {})
        return merged

    def update_config(self, changes: dict) -> None:
        meta = self.meta
        config = dict(meta.get("config") or {})
        config.update({key: value for key, value in changes.items() if value is not None})
        meta["config"] = config
        _write_json(self.meta_path, meta)

    def meetings(self) -> list[Meeting]:
        found = [
            Meeting(project=self, slug=child.name, path=child)
            for child in sorted(self.path.iterdir())
            if child.is_dir() and (child / "meeting.json").exists()
        ]
        return sorted(found, key=lambda meeting: meeting.meta.get("created_at", 0), reverse=True)

    def meeting(self, slug: str) -> Meeting | None:
        path = self.path / slug
        return Meeting(project=self, slug=slug, path=path) if (path / "meeting.json").exists() else None

    def create_meeting(self, name: str, config: dict | None = None) -> Meeting:
        slug = slugify(name)
        candidate = slug
        index = 2
        while (self.path / candidate).exists():
            candidate = f"{slug}-{index}"
            index += 1

        meeting = Meeting(project=self, slug=candidate, path=self.path / candidate)
        meeting.path.mkdir(parents=True, exist_ok=True)
        _write_json(
            meeting.meta_path,
            {"name": name.strip() or candidate, "created_at": time.time(), "config": config or {}},
        )
        return meeting

    def to_json(self) -> dict:
        return {"slug": self.slug, "name": self.name, "config": public_config(self.config)}


@dataclass
class Workspace:
    root: Path
    _ensured: bool = field(default=False, repr=False)

    def ensure(self) -> None:
        if not self._ensured:
            self.root.mkdir(parents=True, exist_ok=True)
            self._ensured = True

    def projects(self) -> list[Project]:
        self.ensure()
        return [
            Project(root=self.root, slug=child.name, path=child)
            for child in sorted(self.root.iterdir())
            if child.is_dir() and (child / "project.json").exists()
        ]

    def project(self, slug: str) -> Project | None:
        path = self.root / slug
        return Project(root=self.root, slug=slug, path=path) if (path / "project.json").exists() else None

    def create_project(self, name: str, config: dict | None = None) -> Project:
        self.ensure()
        slug = slugify(name)
        candidate = slug
        index = 2
        while (self.root / candidate).exists():
            candidate = f"{slug}-{index}"
            index += 1

        project = Project(root=self.root, slug=candidate, path=self.root / candidate)
        project.path.mkdir(parents=True, exist_ok=True)
        _write_json(project.meta_path, {"name": name.strip() or candidate, "config": config or {}})
        return project
