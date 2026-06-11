from __future__ import annotations

import zlib
from pathlib import Path
import struct

GrayImage = list[list[int]]
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


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


def _paeth_predictor(left: int, above: int, upper_left: int) -> int:
    estimate = left + above - upper_left
    left_distance = abs(estimate - left)
    above_distance = abs(estimate - above)
    upper_left_distance = abs(estimate - upper_left)
    if left_distance <= above_distance and left_distance <= upper_left_distance:
        return left
    if above_distance <= upper_left_distance:
        return above
    return upper_left


def _png_channels(color_type: int) -> int:
    channels = {
        0: 1,
        2: 3,
        4: 2,
        6: 4,
    }.get(color_type)
    if channels is None:
        raise ImageFormatError("only grayscale, RGB, GA, and RGBA PNG files are supported")
    return channels


def read_png(path: Path) -> tuple[GrayImage, int, int]:
    data = path.read_bytes()
    if not data.startswith(PNG_SIGNATURE):
        raise ImageFormatError(f"{path} is not a PNG file")

    offset = len(PNG_SIGNATURE)
    width: int | None = None
    height: int | None = None
    bit_depth: int | None = None
    color_type: int | None = None
    compression_method: int | None = None
    filter_method: int | None = None
    interlace_method: int | None = None
    idat_parts: list[bytes] = []

    while offset + 8 <= len(data):
        chunk_length = struct.unpack_from(">I", data, offset)[0]
        chunk_type = data[offset + 4 : offset + 8]
        chunk_start = offset + 8
        chunk_end = chunk_start + chunk_length
        if chunk_end + 4 > len(data):
            raise ImageFormatError("PNG chunk extends past end of file")
        chunk_data = data[chunk_start:chunk_end]
        offset = chunk_end + 4

        if chunk_type == b"IHDR":
            if chunk_length != 13:
                raise ImageFormatError("PNG IHDR chunk has invalid length")
            (
                width,
                height,
                bit_depth,
                color_type,
                compression_method,
                filter_method,
                interlace_method,
            ) = struct.unpack(">IIBBBBB", chunk_data)
        elif chunk_type == b"IDAT":
            idat_parts.append(chunk_data)
        elif chunk_type == b"IEND":
            break

    if width is None or height is None or bit_depth is None or color_type is None:
        raise ImageFormatError("PNG is missing an IHDR chunk")
    if width <= 0 or height <= 0:
        raise ImageFormatError("PNG dimensions must be positive")
    if bit_depth != 8:
        raise ImageFormatError("only 8-bit PNG files are supported")
    if compression_method != 0 or filter_method != 0:
        raise ImageFormatError("unsupported PNG compression or filter method")
    if interlace_method != 0:
        raise ImageFormatError("interlaced PNG files are not supported")
    if not idat_parts:
        raise ImageFormatError("PNG is missing image data")

    channels = _png_channels(color_type)
    stride = width * channels
    try:
        inflated = zlib.decompress(b"".join(idat_parts))
    except zlib.error as exc:
        raise ImageFormatError("PNG image data is not valid zlib data") from exc

    expected = (stride + 1) * height
    if len(inflated) != expected:
        raise ImageFormatError("PNG payload length does not match image dimensions")

    rows: GrayImage = []
    previous = bytearray(stride)
    cursor = 0
    for _ in range(height):
        filter_type = inflated[cursor]
        cursor += 1
        scanline = bytearray(inflated[cursor : cursor + stride])
        cursor += stride

        for index, value in enumerate(scanline):
            left = scanline[index - channels] if index >= channels else 0
            above = previous[index]
            upper_left = previous[index - channels] if index >= channels else 0
            if filter_type == 0:
                predictor = 0
            elif filter_type == 1:
                predictor = left
            elif filter_type == 2:
                predictor = above
            elif filter_type == 3:
                predictor = (left + above) // 2
            elif filter_type == 4:
                predictor = _paeth_predictor(left, above, upper_left)
            else:
                raise ImageFormatError(f"unsupported PNG scanline filter: {filter_type}")
            scanline[index] = (value + predictor) & 0xFF

        row: list[int] = []
        for x in range(width):
            pixel = x * channels
            if color_type == 0:
                row.append(scanline[pixel])
            elif color_type == 4:
                row.append(scanline[pixel])
            else:
                red = scanline[pixel]
                green = scanline[pixel + 1]
                blue = scanline[pixel + 2]
                row.append(round(0.2126 * red + 0.7152 * green + 0.0722 * blue))
        rows.append(row)
        previous = scanline

    return rows, width, height


def read_image(path: Path) -> tuple[GrayImage, int, int]:
    suffix = path.suffix.lower()
    if suffix == ".bmp":
        return read_bmp(path)
    if suffix == ".png":
        return read_png(path)
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
    extensions = {".pgm", ".ppm", ".pnm", ".bmp", ".png"}
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
