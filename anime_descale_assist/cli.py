from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .vapoursynth_backend import VS_HEIGHTS


def _parse_csv_ints(value: str) -> list[int]:
    try:
        return [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _parse_csv_strings(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_csv_floats(value: str) -> list[float]:
    try:
        return [float(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def probe_vs_command(args: argparse.Namespace) -> int:
    from .vapoursynth_backend import probe_samples, vapoursynth_environment

    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.out).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    result = probe_samples(
        input_path,
        heights=args.heights,
        kernels=args.kernels,
        max_samples=args.max_samples,
        score_exponent=args.score_exponent,
        shifts=args.shifts,
    )
    payload = result.to_jsonable()
    payload["vapoursynth"] = vapoursynth_environment()
    _write_json(output_dir / "vs_probe.json", payload)

    print(
        f"best height: {result.best.height}p "
        f"({result.best.width}x{result.best.height}), "
        f"confidence {result.height_confidence:.3f}"
    )
    print(
        f"best kernel: {result.best.kernel}, "
        f"kernel confidence {result.kernel_confidence:.3f}"
    )
    print("height candidates:")
    for candidate in result.height_summary[: args.print_count]:
        print(
            f"  {candidate.width}x{candidate.height} {candidate.kernel:13s} "
            f"score={candidate.score:.6f} raw={candidate.raw_error:.6f} "
            f"shift=({candidate.src_left:g},{candidate.src_top:g}) "
            f"signal={candidate.height_signal:.4f}"
        )
    print("kernel candidates for best height:")
    for candidate in result.kernel_summary[: args.print_count]:
        print(
            f"  {candidate.kernel:13s} "
            f"score={candidate.score:.6f} raw={candidate.raw_error:.6f} "
            f"shift=({candidate.src_left:g},{candidate.src_top:g})"
        )
    print(f"wrote {output_dir / 'vs_probe.json'}")
    return 0


def vs_env_command(args: argparse.Namespace) -> int:
    from .vapoursynth_backend import vapoursynth_environment

    env = vapoursynth_environment()
    print(env["core"])
    print(f"has descale: {env['has_descale']}")
    print("plugins:")
    for plugin in env["plugins"]:
        print(f"  {plugin['namespace']}: {plugin['identifier']} ({plugin['name']})")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="descale-assist",
        description="VapourSynth native-resolution probing for anime descale workflows.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    probe_vs = subparsers.add_parser("probe-vs", help="probe samples with VapourSynth descale")
    probe_vs.add_argument("input", help="sample image file or directory")
    probe_vs.add_argument("--out", required=True, help="output work directory")
    probe_vs.add_argument("--max-samples", type=int, default=8, help="maximum samples to probe")
    probe_vs.add_argument(
        "--heights",
        type=_parse_csv_ints,
        default=list(VS_HEIGHTS),
        help="comma-separated native-height candidates",
    )
    probe_vs.add_argument(
        "--kernels",
        type=_parse_csv_strings,
        default=["bilinear", "catrom", "bicubicsharp", "mitchell", "lanczos3", "spline36"],
        help="comma-separated VapourSynth descale kernels",
    )
    probe_vs.add_argument(
        "--score-exponent",
        type=float,
        default=4.0,
        help="penalty exponent for near-source candidates",
    )
    probe_vs.add_argument(
        "--shifts",
        type=_parse_csv_floats,
        default=[0.0, -0.5, 0.5],
        help="comma-separated src_left/src_top shifts for low-confidence odd-height refinement",
    )
    probe_vs.add_argument("--print-count", type=int, default=8, help="ranked candidates to print")
    probe_vs.set_defaults(func=probe_vs_command)

    vs_env = subparsers.add_parser("vs-env", help="print VapourSynth core and plugin status")
    vs_env.set_defaults(func=vs_env_command)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
