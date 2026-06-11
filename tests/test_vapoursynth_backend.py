from importlib.util import find_spec
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from anime_descale_assist.images import write_pgm
from anime_descale_assist.resample import resize_gray
from anime_descale_assist.vapoursynth_backend import (
    VS_HEIGHTS,
    VSCandidate,
    select_height_candidate,
)


def synthetic_native(width: int, height: int) -> list[list[int]]:
    image: list[list[int]] = []
    for y in range(height):
        row: list[int] = []
        for x in range(width):
            value = 40
            if (x // 4 + y // 3) % 2 == 0:
                value = 190
            if abs(x - y * width // height) <= 1:
                value = 230
            row.append(value)
        image.append(row)
    return image


@unittest.skipUnless(find_spec("vapoursynth"), "VapourSynth is not installed")
class VapourSynthBackendTests(unittest.TestCase):
    def test_probe_samples_finds_synthetic_36p_bilinear(self) -> None:
        from anime_descale_assist.vapoursynth_backend import probe_samples

        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.pgm"
            source = resize_gray(synthetic_native(64, 36), 96, 54, "bilinear")
            write_pgm(path, source)
            result = probe_samples(
                path,
                heights=[36, 45],
                kernels=["bilinear"],
                max_samples=1,
                score_exponent=4,
            )

        self.assertEqual(result.best.height, 36)
        self.assertEqual(result.best.kernel, "bilinear")
        self.assertEqual(result.classification, "high_confidence_height")


class HeightSelectionTests(unittest.TestCase):
    def test_default_vs_candidates_include_800p(self) -> None:
        self.assertIn(800, VS_HEIGHTS)

    def test_selects_first_local_minimum_for_720p_curve(self) -> None:
        summary = [
            VSCandidate(720, 1280, "bilinear", 0.109, 0.553, []),
            VSCandidate(756, 1344, "spline36", 0.141, 0.585, []),
            VSCandidate(810, 1440, "catrom", 0.165, 0.522, []),
            VSCandidate(864, 1536, "bilinear", 0.188, 0.458, []),
            VSCandidate(936, 1664, "bilinear", 0.200, 0.355, []),
        ]
        selected, method = select_height_candidate(summary)
        self.assertEqual(selected.height, 720)
        self.assertEqual(method, "first_local_minimum")

    def test_selects_native_knee_for_864p_curve(self) -> None:
        summary = [
            VSCandidate(720, 1280, "lanczos3", 0.140, 0.708, []),
            VSCandidate(756, 1344, "lanczos3", 0.158, 0.658, []),
            VSCandidate(810, 1440, "lanczos3", 0.169, 0.535, []),
            VSCandidate(864, 1536, "lanczos3", 0.179, 0.437, []),
            VSCandidate(900, 1600, "lanczos3", 0.204, 0.422, []),
            VSCandidate(936, 1664, "lanczos3", 0.207, 0.366, []),
            VSCandidate(1008, 1792, "bilinear", 0.157, 0.206, []),
        ]
        selected, method = select_height_candidate(summary)
        self.assertEqual(selected.height, 864)
        self.assertEqual(method, "native_knee")

    def test_selects_shift_refined_765p_curve(self) -> None:
        summary = [
            VSCandidate(720, 1280, "lanczos3", 0.138, 0.700, []),
            VSCandidate(756, 1344, "lanczos3", 0.156, 0.652, []),
            VSCandidate(765, 1360, "catrom", 0.137, 0.546, [], src_top=-0.5),
            VSCandidate(810, 1440, "spline36", 0.185, 0.585, []),
            VSCandidate(864, 1536, "bicubicsharp", 0.214, 0.522, []),
        ]
        selected, method = select_height_candidate(summary)
        self.assertEqual(selected.height, 765)
        self.assertEqual(method, "native_knee")
        self.assertEqual(selected.src_top, -0.5)


if __name__ == "__main__":
    unittest.main()
