"""Command line entry points.

The dashboard is how AviListener is used: it owns recording, transcription,
speaker splitting and merging, and keeps each meeting's settings with the
meeting. What is left here is the two things that have to work before a
dashboard exists to open - finding out which audio devices this machine has,
and checking that the microphone is actually loud enough to be recorded.
"""

from __future__ import annotations

import argparse
import threading
from pathlib import Path

import yaml
from rich.console import Console
from rich.table import Table

from avilistener.audio import enabled_sources, list_audio_devices

console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(prog="avilistener")
    subparsers = parser.add_subparsers(dest="command", required=True)

    dashboard_parser = subparsers.add_parser("dashboard", help="Run the web dashboard. This is the main interface.")
    dashboard_parser.add_argument("--host", default="127.0.0.1", help="Address to bind (localhost by default).")
    dashboard_parser.add_argument("--port", type=int, default=8000, help="Port to serve on.")

    subparsers.add_parser("list-devices", help="Show audio devices available to AviListener.")

    levels_parser = subparsers.add_parser("levels", help="Measure source audio levels and the gate threshold. Records nothing to disk.")
    levels_parser.add_argument("--config", default="config.yaml", help="Path to config YAML.")
    levels_parser.add_argument("--duration", type=float, default=10.0, help="Seconds to measure for.")

    live_parser = subparsers.add_parser(
        "live",
        help="Prototype: record and transcribe each utterance as it is captured, printing its latency.",
    )
    live_parser.add_argument("--config", default="config.yaml", help="Path to config YAML.")
    live_parser.add_argument("--output", default=None, help="Where to write clips. Defaults to workspace/live/<timestamp>/.")
    live_parser.add_argument(
        "--model",
        default=None,
        help="Whisper model size for this run only. A smaller model (small, medium) lowers latency.",
    )
    live_parser.add_argument("--device", default=None, help="Override the config device (cuda or cpu).")
    live_parser.add_argument("--compute-type", default=None, help="Override the config compute type (for example int8 or float16).")

    args = parser.parse_args()
    if args.command == "dashboard":
        dashboard(args.host, args.port)
        return
    if args.command == "list-devices":
        console.print(list_audio_devices())
        return
    if args.command == "levels":
        levels(Path(args.config), args.duration)
        return
    if args.command == "live":
        live(
            Path(args.config),
            Path(args.output) if args.output else None,
            args.model,
            args.device,
            args.compute_type,
        )
        return


def dashboard(host: str, port: int) -> None:
    try:
        import uvicorn
    except ImportError:
        raise SystemExit('The dashboard needs extra packages. Install them with: pip install -e ".[web]"')

    from avilistener.server.app import WEB_DIST

    if not WEB_DIST.exists():
        console.print("[yellow]The interface has not been built yet. Run:[/yellow]")
        console.print("cd web; npm install; npm run build", soft_wrap=True)
        raise SystemExit(1)

    console.print(f"[green]Dashboard on http://{host}:{port}[/green]")
    uvicorn.run("avilistener.server.app:app", host=host, port=port, log_level="warning")


def levels(config_path: Path, duration: float) -> None:
    """Report what each source is actually hearing.

    A wrong device or a muted input is invisible until a transcript comes back
    empty, which is an expensive way to find out. This measures instead.
    """
    from avilistener.recorder import LevelProbe

    config = _load_config(config_path)
    sources = list(enabled_sources(config.get("sources", {})))
    if not sources:
        raise SystemExit("No enabled sources in config.")

    stop_event = threading.Event()
    console.print(f"[bold]Measuring {', '.join(s.name for s in sources)} for {duration:.0f}s. Speak normally.[/bold]")
    console.print("[cyan]Nothing is written to disk.[/cyan]")

    probes = [LevelProbe(source, _segment_settings(config, source.name), stop_event, duration) for source in sources]
    for probe in probes:
        probe.start()
    for probe in probes:
        probe.join(timeout=duration + 10)

    table = Table(title="Audio levels (RMS)")
    table.add_column("Source", style="cyan")
    table.add_column("Noise floor", justify="right")
    table.add_column("Gate", justify="right")
    table.add_column("Peak", justify="right")
    table.add_column("Above gate", justify="right")
    for probe in probes:
        report = probe.report
        if report.error is not None:
            table.add_row(report.source, "-", "-", "-", f"[red]{report.error}[/red]")
            continue
        headroom = report.peak_block / report.threshold if report.threshold else 0.0
        table.add_row(
            report.source,
            f"{report.noise_floor:.5f}",
            f"{report.threshold:.5f}",
            f"{report.peak_block:.5f}",
            f"{report.voiced_ratio * 100:.0f}%  ({headroom:.0f}x headroom)",
        )
    console.print(table)
    console.print("[cyan]Peak should sit well above the gate while you talk. If it does not,[/cyan]")
    console.print("[cyan]raise your input volume, or set sources.<name>.silence_rms_threshold.[/cyan]")


def live(
    config_path: Path,
    output_dir: Path | None,
    model_size: str | None,
    device: str | None,
    compute_type: str | None,
) -> None:
    """Record and transcribe each utterance as it closes, printing its latency.

    A prototype for one question: on this machine, how long after someone stops
    speaking does their line appear? That number decides whether a live view is
    worth building, and it cannot be answered by reading code.

    Clips are still written to disk in the usual format, so the same session can
    be put through the ordinary pipeline afterwards. Diarization is not
    available live, so lines are labelled by source only.
    """
    from datetime import datetime

    from avilistener.live import LiveTranscriber, format_live_line
    from avilistener.recorder import SourceRecorder
    from avilistener.transcriber import build_transcriber

    config = _load_config(config_path)
    sources = list(enabled_sources(config.get("sources", {})))
    if not sources:
        raise SystemExit("No enabled sources in config.")

    # Copied, never mutated: the loaded config still describes the big model the
    # offline pipeline should use, while live runs can try a faster one.
    effective_config = dict(config)
    if model_size:
        effective_config["model_size"] = model_size
    if device:
        effective_config["device"] = device
    if compute_type:
        effective_config["compute_type"] = compute_type

    if output_dir is None:
        output_dir = Path.cwd() / "workspace" / "live" / datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"[bold]Live transcription prototype[/bold] ({', '.join(s.name for s in sources)})")
    console.print(f"[cyan]Clips are written to {output_dir}[/cyan]")
    console.print(
        f"[cyan]Loading model {effective_config.get('model_size', 'large-v3')} "
        f"on {effective_config.get('device', 'cuda')}. This pause is the model, not recording.[/cyan]"
    )
    # Loaded before any recorder starts, so the load never eats the first
    # utterance and the pause the user sees is explained.
    transcriber = build_transcriber(effective_config)

    def print_line(line) -> None:
        # markup off: a transcript containing square brackets is text, not rich
        # markup, and must not be swallowed or raise.
        console.print(format_live_line(line), markup=False, highlight=False)

    def print_error(path: Path, exc: Exception) -> None:
        console.print(f"Failed to transcribe {path.name}: {exc}", style="red", markup=False, highlight=False)

    live_transcriber = LiveTranscriber(transcriber, print_line, print_error)
    live_transcriber.start()

    stop_event = threading.Event()
    recorders = [
        SourceRecorder(
            source,
            output_dir,
            _segment_settings(config, source.name),
            stop_event,
            on_saved=live_transcriber.submit,
            continuous=False,
        )
        for source in sources
    ]
    for recorder in recorders:
        recorder.start()

    console.print("[green]Listening. Press Ctrl+C to stop.[/green]")
    try:
        while any(recorder.is_alive() for recorder in recorders):
            for recorder in recorders:
                recorder.join(timeout=0.5)
    except KeyboardInterrupt:
        console.print("\n[yellow]Stopping...[/yellow]")

    stop_event.set()
    # Joined rather than abandoned: a recorder killed mid-write leaves a WAV
    # whose header claims zero length.
    for recorder in recorders:
        recorder.join(timeout=30)

    pending = live_transcriber.pending
    if pending:
        console.print(f"[cyan]Transcribing {pending} remaining clip(s)...[/cyan]")
    live_transcriber.stop(drain=True, timeout=300)

    for recorder in recorders:
        if recorder.error is not None:
            console.print(f"[red]{recorder.source.name} failed: {recorder.error}[/red]")

    console.print(
        f"[bold]Transcribed {live_transcriber.lines_emitted} clip(s), "
        f"skipped {live_transcriber.clips_skipped}.[/bold]"
    )
    if live_transcriber.lines_emitted:
        console.print(
            f"[bold]Latency: {live_transcriber.average_latency:.1f}s average, "
            f"{live_transcriber.max_latency:.1f}s worst.[/bold]"
        )
    console.print(f"[cyan]Clips kept in {output_dir} for the offline pipeline.[/cyan]")


def _segment_settings(config: dict, source_name: str):
    """Gate settings for one source.

    Sources differ wildly in level (a quiet microphone versus a speaker
    loopback), so each gets its own gate. Left unset, the gate adapts to that
    source's own noise floor rather than using one number for everything.
    """
    from avilistener.recorder import SegmentSettings

    raw = config.get("recording", {}) or {}
    source_raw = (config.get("sources", {}) or {}).get(source_name, {}) or {}
    threshold = source_raw.get("silence_rms_threshold", raw.get("silence_rms_threshold"))
    return SegmentSettings(
        sample_rate=int(config.get("sample_rate", 16000)),
        silence_rms_threshold=None if threshold in (None, "", "auto") else float(threshold),
        silence_duration_ms=int(raw.get("silence_duration_ms", 1500)),
        preroll_ms=int(raw.get("preroll_ms", 300)),
        min_segment_ms=int(raw.get("min_segment_ms", 700)),
        noise_multiplier=float(raw.get("noise_multiplier", 5.0)),
        min_threshold=float(raw.get("min_threshold", 0.0006)),
    )


def _load_config(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"Config not found: {path}. Copy config.example.yaml to config.yaml first.")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


if __name__ == "__main__":
    main()
