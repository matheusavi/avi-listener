"""Verify an original recordings directory against its backup, storing hashes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from avilistener.timeline import recording_manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("original", type=Path)
    parser.add_argument("backup", type=Path)
    args = parser.parse_args()
    original = recording_manifest(args.original)
    backup = recording_manifest(args.backup)
    if not original or original != backup:
        raise SystemExit("Original and backup recordings differ; verification failed")
    manifest = args.backup.parent / "recording-integrity.json"
    if manifest.exists():
        if json.loads(manifest.read_text(encoding="utf-8")) != original:
            raise SystemExit("Recordings differ from the saved integrity manifest")
    else:
        manifest.write_text(json.dumps(original, indent=2), encoding="utf-8")
    print(f"Verified {len(original)} identical WAVs, {sum(p['bytes'] for p in original)} bytes")
    print(manifest)


if __name__ == "__main__":
    main()
