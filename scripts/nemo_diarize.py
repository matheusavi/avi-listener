from __future__ import annotations

import argparse
import json
import shutil
import wave
from pathlib import Path

from omegaconf import OmegaConf


def main() -> None:
    parser = argparse.ArgumentParser(description="Run NVIDIA NeMo clustering diarization on a WAV file.")
    parser.add_argument("--audio", required=True, help="Path to input WAV/audio file.")
    parser.add_argument("--out", required=True, help="Output directory.")
    parser.add_argument("--num-speakers", type=int, default=None, help="Known number of speakers, if available.")
    parser.add_argument("--max-speakers", type=int, default=8, help="Maximum speakers for clustering.")
    parser.add_argument("--device", default=None, help="Device override, e.g. cuda:0 or cpu.")
    args = parser.parse_args()

    source_audio_path = Path(args.audio).resolve()
    out_dir = Path(args.out).resolve()
    work_dir = out_dir / "nemo_work"
    manifest_path = work_dir / "manifest.json"
    # Start clean. This directory also holds cached embeddings and manifests,
    # and reusing an output directory across runs left stale results behind.
    if work_dir.exists():
        shutil.rmtree(work_dir, ignore_errors=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    audio_path = normalize_audio(source_audio_path, work_dir)
    duration = wav_duration(audio_path)
    manifest = {
        "audio_filepath": str(audio_path),
        "offset": 0,
        "duration": duration,
        "label": "infer",
        "text": "-",
        "num_speakers": args.num_speakers,
        "rttm_filepath": None,
        "uem_filepath": None,
    }
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")

    cfg = OmegaConf.create(
        {
            "name": "ClusterDiarizer",
            "num_workers": 0,
            "sample_rate": 16000,
            "batch_size": 64,
            "device": args.device,
            "verbose": True,
            "diarizer": {
                "manifest_filepath": str(manifest_path),
                "out_dir": str(work_dir),
                "oracle_vad": False,
                "collar": 0.25,
                "ignore_overlap": True,
                "vad": {
                    "model_path": "vad_multilingual_marblenet",
                    "external_vad_manifest": None,
                    "parameters": {
                        "window_length_in_sec": 0.63,
                        "shift_length_in_sec": 0.08,
                        "smoothing": False,
                        "overlap": 0.5,
                        "onset": 0.5,
                        "offset": 0.3,
                        "pad_onset": 0.2,
                        "pad_offset": 0.2,
                        "min_duration_on": 0.35,
                        "min_duration_off": 0.35,
                        "filter_speech_first": True,
                    },
                },
                "speaker_embeddings": {
                    "model_path": "titanet_large",
                    "parameters": {
                        "window_length_in_sec": [1.9, 1.2, 0.5],
                        "shift_length_in_sec": [0.95, 0.6, 0.25],
                        "multiscale_weights": [1, 1, 1],
                        "save_embeddings": True,
                    },
                },
                "clustering": {
                    "parameters": {
                        "oracle_num_speakers": args.num_speakers is not None,
                        "max_num_speakers": args.max_speakers,
                        "enhanced_count_thres": 80,
                        "max_rp_threshold": 0.25,
                        "sparse_search_volume": 10,
                        "maj_vote_spk_count": False,
                        "chunk_cluster_count": 50,
                        "embeddings_per_chunk": 10000,
                    }
                },
                "asr": {
                    "model_path": None,
                    "parameters": {
                        "asr_based_vad": False,
                        "asr_based_vad_threshold": 1.0,
                        "asr_batch_size": None,
                        "decoder_delay_in_sec": None,
                        "word_ts_anchor_offset": None,
                        "word_ts_anchor_pos": "start",
                        "fix_word_ts_with_VAD": False,
                        "colored_text": False,
                        "print_time": True,
                        "break_lines": False,
                    },
                },
            },
        }
    )

    from nemo.collections.asr.models import ClusteringDiarizer

    diarizer = ClusteringDiarizer(cfg=cfg)
    diarizer.diarize()

    # Take the RTTM for THIS audio by name. Globbing and picking the first
    # match silently returned another recording's diarization whenever an
    # output directory was reused, and the transcript looked plausible while
    # every speaker label belonged to a different meeting.
    pred_dir = work_dir / "pred_rttms"
    expected_rttm = pred_dir / f"{audio_path.stem}.rttm"
    if not expected_rttm.exists():
        available = sorted(p.name for p in pred_dir.glob("*.rttm"))
        raise SystemExit(f"No RTTM for {audio_path.stem} under {pred_dir}. Found: {available or 'nothing'}")

    final_rttm = out_dir / f"{source_audio_path.stem}.rttm"
    shutil.copyfile(expected_rttm, final_rttm)
    print(final_rttm)


def normalize_audio(source_path: Path, work_dir: Path) -> Path:
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly

    data, sample_rate = sf.read(str(source_path), always_2d=True, dtype="float32")
    mono = data.mean(axis=1)
    target_rate = 16000
    if sample_rate != target_rate:
        gcd = int(np.gcd(sample_rate, target_rate))
        mono = resample_poly(mono, target_rate // gcd, sample_rate // gcd).astype("float32")
    normalized_path = work_dir / f"{source_path.stem}.mono16k.wav"
    sf.write(str(normalized_path), mono, target_rate, subtype="PCM_16")
    return normalized_path


def wav_duration(path: Path) -> float | None:
    if path.suffix.lower() != ".wav":
        return None
    with wave.open(str(path), "rb") as f:
        return f.getnframes() / float(f.getframerate())


if __name__ == "__main__":
    main()
