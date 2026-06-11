import unittest

from anime_descale_assist.clustering import cluster_samples
from anime_descale_assist.resample import resize_gray
from anime_descale_assist.scoring import analyze_image


def synthetic_native(width: int, height: int) -> list[list[int]]:
    image: list[list[int]] = []
    for y in range(height):
        row: list[int] = []
        for x in range(width):
            value = 48
            if (x // 5 + y // 3) % 2 == 0:
                value = 196
            if abs(x - y * width // height) <= 1:
                value = 240
            row.append(value)
        image.append(row)
    return image


class ScoringTests(unittest.TestCase):
    def test_roundtrip_prefers_true_synthetic_height(self) -> None:
        native = synthetic_native(80, 45)
        source = resize_gray(native, 160, 90, "bilinear")
        result, _preview = analyze_image(
            image=source,
            sample_file="synthetic.pgm",
            sample_index=0,
            timestamp_seconds=0.0,
            original_width=160,
            original_height=90,
            candidate_heights=[45, 60, 72],
            kernels=["bilinear"],
            max_analysis_width=160,
        )
        self.assertIsNotNone(result.best)
        self.assertEqual(result.best.height, 45)
        self.assertEqual(result.best.kernel, "bilinear")

    def test_cluster_samples_groups_by_label(self) -> None:
        native = synthetic_native(80, 45)
        source = resize_gray(native, 160, 90, "bilinear")
        result, _preview = analyze_image(
            image=source,
            sample_file="synthetic.pgm",
            sample_index=0,
            timestamp_seconds=0.0,
            original_width=160,
            original_height=90,
            candidate_heights=[45, 60],
            kernels=["bilinear"],
            max_analysis_width=160,
        )
        clusters = cluster_samples([result])
        self.assertEqual(len(clusters), 1)
        self.assertEqual(clusters[0].count, 1)


if __name__ == "__main__":
    unittest.main()
