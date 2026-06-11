from __future__ import annotations

import math
from functools import lru_cache

from .images import GrayImage


def _sinc(x: float) -> float:
    if abs(x) < 1e-8:
        return 1.0
    pix = math.pi * x
    return math.sin(pix) / pix


def _mitchell_netravali(x: float, b: float, c: float) -> float:
    x = abs(x)
    if x < 1:
        return (
            (12 - 9 * b - 6 * c) * x**3
            + (-18 + 12 * b + 6 * c) * x**2
            + (6 - 2 * b)
        ) / 6
    if x < 2:
        return (
            (-b - 6 * c) * x**3
            + (6 * b + 30 * c) * x**2
            + (-12 * b - 48 * c) * x
            + (8 * b + 24 * c)
        ) / 6
    return 0.0


def kernel_support(name: str) -> float:
    normalized = normalize_kernel_name(name)
    if normalized == "bilinear":
        return 1.0
    if normalized in {"catrom", "bicubicsharp", "mitchell"}:
        return 2.0
    if normalized == "lanczos2":
        return 2.0
    if normalized == "lanczos3":
        return 3.0
    raise ValueError(f"unknown kernel: {name}")


def normalize_kernel_name(name: str) -> str:
    return name.strip().lower().replace("-", "").replace("_", "")


def kernel_value(name: str, x: float) -> float:
    normalized = normalize_kernel_name(name)
    ax = abs(x)
    if normalized == "bilinear":
        return max(0.0, 1.0 - ax)
    if normalized == "catrom":
        return _mitchell_netravali(x, b=0.0, c=0.5)
    if normalized == "bicubicsharp":
        return _mitchell_netravali(x, b=0.0, c=0.75)
    if normalized == "mitchell":
        return _mitchell_netravali(x, b=1 / 3, c=1 / 3)
    if normalized == "lanczos2":
        if ax >= 2:
            return 0.0
        return _sinc(x) * _sinc(x / 2)
    if normalized == "lanczos3":
        if ax >= 3:
            return 0.0
        return _sinc(x) * _sinc(x / 3)
    raise ValueError(f"unknown kernel: {name}")


@lru_cache(maxsize=256)
def _contributions(
    in_size: int, out_size: int, kernel_name: str
) -> tuple[tuple[tuple[int, float], ...], ...]:
    if in_size <= 0 or out_size <= 0:
        raise ValueError("image dimensions must be positive")
    if in_size == out_size:
        return tuple(((i, 1.0),) for i in range(out_size))

    scale = in_size / out_size
    filter_scale = max(1.0, scale)
    support = kernel_support(kernel_name) * filter_scale
    rows: list[tuple[tuple[int, float], ...]] = []

    for out_index in range(out_size):
        center = (out_index + 0.5) * scale - 0.5
        left = math.floor(center - support)
        right = math.ceil(center + support)
        weights: list[tuple[int, float]] = []
        total = 0.0

        for src_index in range(left, right + 1):
            clamped = min(in_size - 1, max(0, src_index))
            weight = kernel_value(kernel_name, (center - src_index) / filter_scale)
            if weight == 0:
                continue
            weights.append((clamped, weight))
            total += weight

        if not weights or abs(total) < 1e-12:
            nearest = min(in_size - 1, max(0, round(center)))
            rows.append(((nearest, 1.0),))
            continue

        normalized: dict[int, float] = {}
        for src_index, weight in weights:
            normalized[src_index] = normalized.get(src_index, 0.0) + weight / total
        rows.append(tuple(sorted(normalized.items())))

    return tuple(rows)


def resize_gray(image: GrayImage, out_width: int, out_height: int, kernel: str) -> GrayImage:
    if not image or not image[0]:
        raise ValueError("cannot resize an empty image")
    in_height = len(image)
    in_width = len(image[0])
    if out_width <= 0 or out_height <= 0:
        raise ValueError("output dimensions must be positive")
    if in_width == out_width and in_height == out_height:
        return [row[:] for row in image]

    kernel_name = normalize_kernel_name(kernel)
    horizontal = _contributions(in_width, out_width, kernel_name)
    tmp: list[list[float]] = []
    for row in image:
        out_row: list[float] = []
        for weights in horizontal:
            value = 0.0
            for src_x, weight in weights:
                value += row[src_x] * weight
            out_row.append(value)
        tmp.append(out_row)

    vertical = _contributions(in_height, out_height, kernel_name)
    out: GrayImage = []
    for weights in vertical:
        row: list[int] = []
        for x in range(out_width):
            value = 0.0
            for src_y, weight in weights:
                value += tmp[src_y][x] * weight
            row.append(max(0, min(255, round(value))))
        out.append(row)
    return out

