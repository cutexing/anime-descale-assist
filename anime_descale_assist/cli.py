from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .clustering import build_zones, cluster_samples
from .emit_vpy import emit_vpy
from .ffmpeg import FFmpegError, extract_sparse_samples
from .images import list_sample_images, read_image, triptych, write_pgm
from .report import load_analysis, load_zones, write_reports
from .scoring import DEFAULT_HEIGHTS, DEFAULT_KERNELS, analyze_image
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


def _sample_paths(
    input_path: Path,
    output_dir: Path,
    no_extract: bool,
    interval: float,
    max_samples: int,
) -> list[Path]:
    if input_path.is_dir():
        paths = list_sample_images(input_path)
        if not paths:
            raise SystemExit(f"no .pgm/.ppm/.pnm/.bmp samples found in {input_path}")
        return paths[:max_samples]

    if no_extract:
        raise SystemExit("--no-extract requires INPUT to be a sample directory")

    sample_dir = output_dir / "samples"
    try:
        paths = extract_sparse_samples(input_path, sample_dir, interval, max_samples)
    except FFmpegError as exc:
        raise SystemExit(f"sample extraction failed: {exc}") from exc
    if not paths:
        raise SystemExit("ffmpeg produced no samples")
    return paths


def analyze_command(args: argparse.Namespace) -> int:
    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.out).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_paths = _sample_paths(
        input_path,
        output_dir,
        args.no_extract,
        args.sample_interval,
        args.max_samples,
    )
    preview_dir = output_dir / "preview"
    sample_results = []

    for index, sample_path in enumerate(sample_paths):
        image, width, height = read_image(sample_path)
        timestamp = index * args.sample_interval if not input_path.is_dir() else None
        result, reconstruction = analyze_image(
            image=image,
            sample_file=str(sample_path),
            sample_index=index,
            timestamp_seconds=timestamp,
            original_width=width,
            original_height=height,
            candidate_heights=args.candidates,
            kernels=args.kernels,
            max_analysis_width=args.max_analysis_width,
        )
        sample_results.append(result)

        if reconstruction is not None and index < args.preview_count:
            preview_source = image
            if width != result.analysis_width or height != result.analysis_height:
                from .resample import resize_gray

                preview_source = resize_gray(
                    image,
                    result.analysis_width,
                    result.analysis_height,
                    "bilinear",
                )
            preview = triptych(preview_source, reconstruction)
            write_pgm(preview_dir / f"qc_{index:04d}_{result.label}.pgm", preview)

    clusters = cluster_samples(sample_results)
    zones = build_zones(
        sample_results,
        None if input_path.is_dir() else args.sample_interval,
    )
    write_reports(
        output_dir,
        str(input_path),
        None if input_path.is_dir() else args.sample_interval,
        args.candidates,
        args.kernels,
        sample_results,
        clusters,
        zones,
    )

    print(f"analyzed {len(sample_results)} samples")
    print(f"wrote {output_dir / 'analysis.json'}")
    print(f"wrote {output_dir / 'zones.json'}")
    print(f"wrote {output_dir / 'summary.md'}")
    if clusters:
        top = clusters[0]
        print(
            f"top cluster: {top.label} ({top.count} samples, "
            f"{top.percentage:.2f}%, confidence {top.median_confidence:.3f})"
        )
    return 0


def qc_command(args: argparse.Namespace) -> int:
    work_dir = Path(args.workdir).expanduser().resolve()
    analysis = load_analysis(work_dir)
    zones = load_zones(work_dir)
    print(f"source: {analysis.get('source')}")
    print(f"samples: {len(analysis.get('samples', []))}")
    print("clusters:")
    for cluster in analysis.get("clusters", []):
        print(
            f"  {cluster['label']}: {cluster['count']} samples, "
            f"{cluster['percentage']:.2f}%, confidence {cluster['median_confidence']:.3f}, "
            f"class {cluster['classification']}"
        )
    print("zones:")
    for zone in zones.get("zones", []):
        seconds = ""
        if zone.get("start_seconds") is not None and zone.get("end_seconds") is not None:
            seconds = f" {zone['start_seconds']:.2f}-{zone['end_seconds']:.2f}s"
        print(
            f"  samples {zone['sample_start']}-{zone['sample_end']}{seconds}: "
            f"{zone['label']} -> {zone['action']} ({zone['confidence']:.3f})"
        )
    return 0


def emit_vpy_command(args: argparse.Namespace) -> int:
    work_dir = Path(args.workdir).expanduser().resolve()
    zones = load_zones(work_dir)
    source = args.source or zones.get("source")
    if not source:
        raise SystemExit("source path missing; pass --source")
    output_path = Path(args.out).expanduser().resolve()
    emit_vpy(source, zones, output_path, fps=args.fps)
    print(f"wrote {output_path}")
    return 0


def probe_vs_command(args: argparse.Namespace) -> int:
    from .report import write_json
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
    write_json(output_dir / "vs_probe.json", payload)

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
        description="Conservative native-resolution auditing for anime descale workflows.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze = subparsers.add_parser("analyze", help="sample and analyze a video or image sample directory")
    analyze.add_argument("input", help="video file or directory of .pgm/.ppm/.pnm/.bmp samples")
    analyze.add_argument("--out", required=True, help="output work directory")
    analyze.add_argument("--sample-interval", type=float, default=5.0, help="seconds between sparse samples")
    analyze.add_argument("--max-samples", type=int, default=120, help="maximum samples to analyze")
    analyze.add_argument("--max-analysis-width", type=int, default=480, help="downscaled width for fast scoring")
    analyze.add_argument("--preview-count", type=int, default=12, help="number of QC triptychs to write")
    analyze.add_argument("--no-extract", action="store_true", help="treat INPUT as an existing sample directory")
    analyze.add_argument(
        "--candidates",
        type=_parse_csv_ints,
        default=list(DEFAULT_HEIGHTS),
        help="comma-separated native-height candidates",
    )
    analyze.add_argument(
        "--kernels",
        type=_parse_csv_strings,
        default=list(DEFAULT_KERNELS),
        help="comma-separated resize kernels",
    )
    analyze.set_defaults(func=analyze_command)

    qc = subparsers.add_parser("qc", help="print a compact report for a work directory")
    qc.add_argument("workdir")
    qc.set_defaults(func=qc_command)

    vpy = subparsers.add_parser("emit-vpy", help="emit a starter VapourSynth script")
    vpy.add_argument("workdir")
    vpy.add_argument("--out", required=True)
    vpy.add_argument("--source", help="override source path written in zones.json")
    vpy.add_argument("--fps", default="24000/1001", help="timeline FPS for converting seconds to frames")
    vpy.set_defaults(func=emit_vpy_command)

    probe_vs = subparsers.add_parser("probe-vs", help="probe samples with VapourSynth descale")
    probe_vs.add_argument("input", help="sample image file or directory")
    probe_vs.add_argument("--out", required=True, help="output work directory")
    probe_vs.add_argument("--max-samples", type=int, default=8, help="maximum samples to analyze")
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
