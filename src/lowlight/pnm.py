"""Netpbm (PGM/PPM) reading and writing, so real images need no dependency.

Pillow would be one line. It is also 3 MB of native code to read a format whose
entire specification is a magic number, three integers and a byte array — and
the repository's claim is about what happens to the bytes, so the bytes are
read here.

Supported: P2/P5 (grey) and P3/P6 (RGB), maxval 255 only. A 16-bit file is
rejected rather than silently truncated, because truncating it would change the
distinct-level count that every measurement in this repository turns on.
"""

from __future__ import annotations

from pathlib import Path

from lowlight.raster import MAX_VALUE, Raster, RasterError

MAGIC_CHANNELS = {b"P2": 1, b"P5": 1, b"P3": 3, b"P6": 3}
BINARY_MAGIC = {b"P5", b"P6"}


class PnmError(RasterError):
    """The file is not a Netpbm image this module can read."""


def loads(payload: bytes) -> Raster:
    """Parse a PGM/PPM byte string."""
    magic = payload[:2]
    if magic not in MAGIC_CHANNELS:
        raise PnmError(
            f"unsupported magic {magic!r}; expected one of "
            f"{sorted(m.decode() for m in MAGIC_CHANNELS)}"
        )
    channels = MAGIC_CHANNELS[magic]

    header, offset = _read_header(payload)
    width, height, maxval = header
    if maxval != MAX_VALUE:
        raise PnmError(f"only 8-bit files are supported; maxval is {maxval}")

    count = width * height * channels
    if magic in BINARY_MAGIC:
        body = payload[offset : offset + count]
        if len(body) != count:
            raise PnmError(f"truncated raster: expected {count} bytes, got {len(body)}")
        data = tuple(body)
    else:
        tokens = payload[offset:].split()
        if len(tokens) < count:
            raise PnmError(
                f"truncated raster: expected {count} samples, got {len(tokens)}"
            )
        data = tuple(int(token) for token in tokens[:count])
    return Raster(width, height, channels, data)


def _read_header(payload: bytes) -> tuple[tuple[int, int, int], int]:
    """Three integers after the magic, skipping whitespace and # comments."""
    values: list[int] = []
    index = 2
    while len(values) < 3:
        if index >= len(payload):
            raise PnmError("header ended before width, height and maxval were read")
        byte = payload[index : index + 1]
        if byte == b"#":
            while index < len(payload) and payload[index : index + 1] not in (
                b"\n",
                b"\r",
            ):
                index += 1
            continue
        if byte.isspace():
            index += 1
            continue
        start = index
        while index < len(payload) and not payload[index : index + 1].isspace():
            index += 1
        token = payload[start:index]
        if not token.isdigit():
            raise PnmError(f"expected an integer in the header; got {token!r}")
        values.append(int(token))
    # Exactly one whitespace byte separates the header from a binary raster.
    return (values[0], values[1], values[2]), index + 1


def dumps(raster: Raster, *, binary: bool = True) -> bytes:
    """Serialise to PGM/PPM."""
    magic = {(1, True): b"P5", (1, False): b"P2", (3, True): b"P6", (3, False): b"P3"}[
        (raster.channels, binary)
    ]
    header = b"%s\n%d %d\n%d\n" % (magic, raster.width, raster.height, MAX_VALUE)
    if binary:
        return header + bytes(raster.data)
    rows = []
    stride = raster.width * raster.channels
    for y in range(raster.height):
        row = raster.data[y * stride : (y + 1) * stride]
        rows.append(b" ".join(b"%d" % v for v in row))
    return header + b"\n".join(rows) + b"\n"


def read(path: str | Path) -> Raster:
    return loads(Path(path).read_bytes())


def write(path: str | Path, raster: Raster, *, binary: bool = True) -> None:
    Path(path).write_bytes(dumps(raster, binary=binary))
