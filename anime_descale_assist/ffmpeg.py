from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path


class FFmpegError(RuntimeError):
    pass


def require_tool(name: str) -> str:
    path = shutil.which(name)
    if path:
        return path

    if name == "ffmpeg":
        try:
            import imageio_ffmpeg
        except ImportError as exc:
            raise FFmpegError(
                "ffmpeg was not found on PATH and imageio-ffmpeg is not installed"
            ) from exc
        return imageio_ffmpeg.get_ffmpeg_exe()

    raise FFmpegError(f"{name} was not found on PATH")


def probe_video(path: Path) -> dict[str, object]:
    ffprobe = require_tool("ffprobe")
    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,r_frame_rate,duration",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise FFmpegError(result.stderr.strip() or "ffprobe failed")
    return json.loads(result.stdout)


def extract_sparse_samples(
    input_path: Path,
    output_dir: Path,
    interval_seconds: float,
    max_samples: int,
) -> list[Path]:
    if interval_seconds <= 0:
        raise ValueError("interval_seconds must be positive")
    if max_samples <= 0:
        raise ValueError("max_samples must be positive")

    ffmpeg = require_tool("ffmpeg")
    output_dir.mkdir(parents=True, exist_ok=True)
    pattern = output_dir / "sample_%06d.pgm"
    fps = f"1/{interval_seconds:g}"
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(input_path),
        "-vf",
        f"fps={fps},format=gray",
        "-frames:v",
        str(max_samples),
        str(pattern),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise FFmpegError(result.stderr.strip() or "ffmpeg sample extraction failed")
    return sorted(output_dir.glob("sample_*.pgm"))
