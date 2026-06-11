from importlib.util import find_spec
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from anime_descale_assist.images import write_pgm
from anime_descale_assist.resample import resize_gray


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


if __name__ == "__main__":
    unittest.main()
