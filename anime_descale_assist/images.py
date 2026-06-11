from __future__ import annotations

from pathlib import Path
import struct

GrayImage = list[list[int]]


class ImageFormatError(ValueError):
    pass


def _read_token(data: bytes, offset: int) -> tuple[bytes, int]:
    length = len(data)
    while offset < length:
        byte = data[offset]
        if byte == ord("#"):
            while offset < length and data[offset] not in (10, 13):
                offset += 1
        elif chr(byte).isspace():
            offset += 1
        else:
            break

    start = offset
    while offset < length and not chr(data[offset]).isspace():
        offset += 1

    if start == offset:
        raise ImageFormatError("unexpected end of PNM header")

    return data[start:offset], offset


def read_pnm(path: Path) -> tuple[GrayImage, int, int]:
    data = path.read_bytes()
    magic, offset = _read_token(data, 0)
    if magic not in (b"P5", b"P6"):
        raise ImageFormatError(f"{path} is not a binary PGM/PPM file")

    width_token, offset = _read_token(data, offset)
    height_token, offset = _read_token(data, offset)
    maxval_token, offset = _read_token(data, offset)

    width = int(width_token)
    height = int(height_token)
    maxval = int(maxval_token)
    if width <= 0 or height <= 0:
        raise ImageFormatError("image dimensions must be positive")
    if maxval <= 0 or maxval > 255:
        raise ImageFormatError("only 8-bit PNM images are supported")

    if offset >= len(data) or not chr(data[offset]).isspace():
        raise ImageFormatError("PNM header is missing a raster separator")
    if data[offset] == 13 and offset + 1 < len(data) and data[offset + 1] == 10:
        offset += 2
    else:
        offset += 1

    if magic == b"P5":
        expected = width * height
        payload = data[offset : offset + expected]
        if len(payload) != expected:
            raise ImageFormatError("PGM payload is shorter than expected")
        rows = [
            list(payload[row * width : (row + 1) * width])
            for row in range(height)
        ]
        return rows, width, height

    expected = width * height * 3
    payload = data[offset : offset + expected]
    if len(payload) != expected:
        raise ImageFormatError("PPM payload is shorter than expected")

    rows: GrayImage = []
    cursor = 0
    for _ in range(height):
        row: list[int] = []
        for _ in range(width):
            red = payload[cursor]
            green = payload[cursor + 1]
            blue = payload[cursor + 2]
            cursor += 3
            row.append(round(0.2126 * red + 0.7152 * green + 0.0722 * blue))
        rows.append(row)
    return rows, width, height


def read_bmp(path: Path) -> tuple[GrayImage, int, int]:
    data = path.read_bytes()
    if len(data) < 54 or data[:2] != b"BM":
        raise ImageFormatError(f"{path} is not a BMP file")

    pixel_offset = struct.unpack_from("<I", data, 10)[0]
    dib_size = struct.unpack_from("<I", data, 14)[0]
    if dib_size < 40:
        raise ImageFormatError("only BITMAPINFOHEADER-style BMP files are supported")

    width = struct.unpack_from("<i", data, 18)[0]
    signed_height = struct.unpack_from("<i", data, 22)[0]
    planes = struct.unpack_from("<H", data, 26)[0]
    bits_per_pixel = struct.unpack_from("<H", data, 28)[0]
    compression = struct.unpack_from("<I", data, 30)[0]

    if planes != 1:
        raise ImageFormatError("invalid BMP plane count")
    if width <= 0 or signed_height == 0:
        raise ImageFormatError("BMP dimensions must be positive")
    if bits_per_pixel not in (24, 32):
        raise ImageFormatError("only 24-bit and 32-bit BMP files are supported")
    if compression != 0:
        raise ImageFormatError("compressed BMP files are not supported")

    height = abs(signed_height)
    top_down = signed_height < 0
    bytes_per_pixel = bits_per_pixel // 8
    stride = ((width * bytes_per_pixel + 3) // 4) * 4
    required = pixel_offset + stride * height
    if len(data) < required:
        raise ImageFormatError("BMP payload is shorter than expected")

    rows: GrayImage = []
    for out_y in range(height):
        src_y = out_y if top_down else height - 1 - out_y
        row_offset = pixel_offset + src_y * stride
        row: list[int] = []
        for x in range(width):
            cursor = row_offset + x * bytes_per_pixel
            blue = data[cursor]
            green = data[cursor + 1]
            red = data[cursor + 2]
            row.append(round(0.2126 * red + 0.7152 * green + 0.0722 * blue))
        rows.append(row)
    return rows, width, height


def read_image(path: Path) -> tuple[GrayImage, int, int]:
    if path.suffix.lower() == ".bmp":
        return read_bmp(path)
    return read_pnm(path)


def write_pgm(path: Path, image: GrayImage) -> None:
    if not image or not image[0]:
        raise ImageFormatError("cannot write an empty image")

    height = len(image)
    width = len(image[0])
    payload = bytearray()
    for row in image:
        if len(row) != width:
            raise ImageFormatError("image rows have inconsistent widths")
        payload.extend(max(0, min(255, int(value))) for value in row)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(f"P5\n{width} {height}\n255\n".encode("ascii") + payload)


def list_sample_images(path: Path) -> list[Path]:
    extensions = {".pgm", ".ppm", ".pnm", ".bmp"}
    return sorted(
        item
        for item in path.rglob("*")
        if item.is_file() and item.suffix.lower() in extensions
    )


def fit_width(width: int, height: int, max_width: int) -> tuple[int, int]:
    if max_width <= 0 or width <= max_width:
        return width, height
    new_width = max(16, max_width)
    new_height = max(16, round(height * (new_width / width)))
    return new_width, new_height


def triptych(source: GrayImage, reconstruction: GrayImage, diff_gain: float = 6.0) -> GrayImage:
    height = min(len(source), len(reconstruction))
    width = min(len(source[0]), len(reconstruction[0]))
    out: GrayImage = []
    gutter = [24] * 4
    for y in range(height):
        src_row = source[y][:width]
        rec_row = reconstruction[y][:width]
        diff_row = [
            max(0, min(255, round(abs(src_row[x] - rec_row[x]) * diff_gain)))
            for x in range(width)
        ]
        out.append(src_row + gutter + rec_row + gutter + diff_row)
    return out
