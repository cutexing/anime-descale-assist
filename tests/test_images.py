from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import struct
import zlib

from anime_descale_assist.images import list_sample_images, read_bmp, read_png, read_pnm, write_pgm


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    checksum = zlib.crc32(kind + payload) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", checksum)


def _write_rgb_png(path: Path) -> None:
    width = 2
    height = 2
    pixels = bytes(
        [
            0,
            255,
            0,
            0,
            0,
            255,
            0,
            0,
            0,
            0,
            255,
            255,
            255,
            255,
        ]
    )
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk("IHDR".encode("ascii"), struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + _png_chunk("IDAT".encode("ascii"), zlib.compress(pixels))
        + _png_chunk("IEND".encode("ascii"), b"")
    )


class ImageTests(unittest.TestCase):
    def test_pgm_roundtrip(self) -> None:
        image = [
            [0, 64, 128],
            [255, 32, 16],
        ]
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.pgm"
            write_pgm(path, image)
            decoded, width, height = read_pnm(path)
        self.assertEqual(width, 3)
        self.assertEqual(height, 2)
        self.assertEqual(decoded, image)

    def test_pgm_preserves_leading_whitespace_pixel_values(self) -> None:
        image = [
            [10, 32, 9],
            [13, 64, 255],
        ]
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.pgm"
            write_pgm(path, image)
            decoded, width, height = read_pnm(path)
        self.assertEqual((width, height), (3, 2))
        self.assertEqual(decoded, image)

    def test_read_24_bit_bmp(self) -> None:
        width = 2
        height = 2
        row_stride = 8
        pixel_data = bytes(
            [
                255, 0, 0, 255, 255, 255, 0, 0,
                0, 0, 255, 0, 255, 0, 0, 0,
            ]
        )
        file_size = 54 + len(pixel_data)
        header = (
            b"BM"
            + struct.pack("<IHHI", file_size, 0, 0, 54)
            + struct.pack(
                "<IiiHHIIiiII",
                40,
                width,
                height,
                1,
                24,
                0,
                row_stride * height,
                0,
                0,
                0,
                0,
            )
        )
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.bmp"
            path.write_bytes(header + pixel_data)
            decoded, decoded_width, decoded_height = read_bmp(path)
        self.assertEqual((decoded_width, decoded_height), (2, 2))
        self.assertEqual(decoded, [[54, 182], [18, 255]])

    def test_read_8_bit_rgb_png(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.png"
            _write_rgb_png(path)
            decoded, width, height = read_png(path)
        self.assertEqual((width, height), (2, 2))
        self.assertEqual(decoded, [[54, 182], [18, 255]])

    def test_list_sample_images_is_recursive(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            nested = root / "nested"
            nested.mkdir()
            write_pgm(nested / "frame.pgm", [[0]])
            _write_rgb_png(nested / "frame.png")
            self.assertEqual(list_sample_images(root), [nested / "frame.pgm", nested / "frame.png"])


if __name__ == "__main__":
    unittest.main()
