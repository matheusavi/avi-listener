"""Run pyannote speaker diarization on a WAV file.

Runs in .venv-pyannote, not the main venv, for the same reason NeMo has its
own: torch-heavy dependencies that conflict with everything else. The contract
mirrors scripts/nemo_diarize.py - write `<audio stem>.rttm` into --out and
print its path as the last line - so meeting.py can drive either engine.

The model is gated on Hugging Face: accept its conditions on the model page
once, then provide a token via --hf-token or the HF_TOKEN environment
variable. Without one this exits with instructions rather than a stack trace.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

MODEL = "pyannote/speaker-diarization-community-1"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run pyannote diarization on a WAV file.")
    parser.add_argument("--audio", required=True, help="Path to input WAV/audio file.")
    parser.add_argument("--out", required=True, help="Output directory.")
    parser.add_argument("--num-speakers", type=int, default=None, help="Known number of speakers, if available.")
    parser.add_argument("--max-speakers", type=int, default=None, help="Upper bound when the count is unknown.")
    parser.add_argument("--model", default=MODEL, help="Hugging Face pipeline to load.")
    parser.add_argument("--hf-token", default=None, help="Hugging Face token; defaults to HF_TOKEN.")
    parser.add_argument("--device", default=None, help="Device override, e.g. cuda:0 or cpu.")
    args = parser.parse_args()

    token = args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise SystemExit(
            f"{args.model} is gated. Create a token at https://huggingface.co/settings/tokens, "
            f"accept the conditions at https://huggingface.co/{args.model}, "
            "and set the HF_TOKEN environment variable."
        )

    audio_path = Path(args.audio).resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    import torch
    from pyannote.audio import Pipeline

    pipeline = Pipeline.from_pretrained(args.model, token=token)
    if pipeline is None:
        raise SystemExit(
            f"Could not load {args.model}. Most likely the conditions at "
            f"https://huggingface.co/{args.model} have not been accepted for this token."
        )
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    pipeline.to(torch.device(device))

    kwargs = {}
    if args.num_speakers is not None:
        kwargs["num_speakers"] = args.num_speakers
    elif args.max_speakers is not None:
        kwargs["max_speakers"] = args.max_speakers

    # Hand the pipeline an in-memory waveform instead of a path: pyannote 4
    # decodes files through torchcodec, which on Windows wants FFmpeg shared
    # DLLs that are usually absent. Our input is always a plain WAV, so the
    # stdlib reads it fine and no decoder is involved.
    import numpy as np
    import wave

    with wave.open(str(audio_path), "rb") as reader:
        sample_rate = reader.getframerate()
        channels = reader.getnchannels()
        frames = np.frombuffer(reader.readframes(reader.getnframes()), dtype=np.int16)
    samples = frames.astype(np.float32) / 32768.0
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    waveform = torch.from_numpy(samples).unsqueeze(0)

    print(f"Diarizing {audio_path.name} on {device} with {args.model}", flush=True)
    result = pipeline({"waveform": waveform, "sample_rate": sample_rate}, **kwargs)
    # Newer pipelines return a result object wrapping the annotation.
    annotation = getattr(result, "speaker_diarization", result)

    final_rttm = out_dir / f"{audio_path.stem}.rttm"
    with open(final_rttm, "w", encoding="utf-8") as sink:
        annotation.write_rttm(sink)
    print(final_rttm)


if __name__ == "__main__":
    main()
