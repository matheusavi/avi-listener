"""Assemble the frames record.mjs captured into one GIF per flow.

    python e2e/demo/make_gif.py <frames-dir> <flow> <output.gif> [width]

Needs Pillow. Frames share one palette so the picture does not flicker as it
plays, and are scaled down because a README GIF at full width is many megabytes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image


def main() -> None:
    frames_dir, flow, output = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
    width = int(sys.argv[4]) if len(sys.argv) > 4 else 960

    manifest = json.loads((frames_dir / f"{flow}.json").read_text(encoding="utf-8"))
    images = []
    for entry in manifest:
        image = Image.open(frames_dir / flow / entry["file"]).convert("RGB")
        height = round(image.height * width / image.width)
        images.append(image.resize((width, height), Image.LANCZOS))

    # One palette for the whole clip, built from a sample of frames.
    sample = Image.new("RGB", (width, images[0].height * min(6, len(images))))
    for index, image in enumerate(images[:: max(1, len(images) // 6)][:6]):
        sample.paste(image, (0, index * image.height))
    palette = sample.quantize(colors=200, method=Image.Quantize.MEDIANCUT)
    quantized = [image.quantize(palette=palette, dither=Image.Dither.NONE) for image in images]

    output.parent.mkdir(parents=True, exist_ok=True)
    quantized[0].save(
        output,
        save_all=True,
        append_images=quantized[1:],
        duration=[entry["duration"] for entry in manifest],
        loop=0,
        optimize=True,
        disposal=1,
    )
    total = sum(entry["duration"] for entry in manifest) / 1000
    print(f"{output}: {len(quantized)} frames, {total:.1f}s, {output.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
