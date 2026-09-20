"""Netpbm round trips, and the files that must be refused."""

import pytest

from lowlight.pnm import PnmError, dumps, loads, read, write
from lowlight.raster import grey, rgb
from lowlight.synthetic import pattern


def test_binary_grey_round_trip():
    image = grey(3, 2, [0, 10, 20, 30, 40, 255])
    assert loads(dumps(image)).data == image.data


def test_ascii_grey_round_trip():
    image = grey(3, 2, [0, 10, 20, 30, 40, 255])
    assert loads(dumps(image, binary=False)).data == image.data


def test_binary_rgb_round_trip():
    image = rgb(2, 1, [1, 2, 3, 250, 251, 252])
    restored = loads(dumps(image))
    assert restored.channels == 3
    assert restored.data == image.data


def test_ascii_rgb_round_trip():
    image = rgb(2, 1, [1, 2, 3, 250, 251, 252])
    assert loads(dumps(image, binary=False)).data == image.data


def test_a_large_pattern_survives_the_round_trip():
    image = pattern(32)
    assert loads(dumps(image)).data == image.data


def test_geometry_survives_the_round_trip():
    image = grey(7, 5, [0] * 35)
    restored = loads(dumps(image))
    assert (restored.width, restored.height) == (7, 5)


def test_magic_reflects_channels_and_encoding():
    assert dumps(grey(1, 1, [0])).startswith(b"P5")
    assert dumps(grey(1, 1, [0]), binary=False).startswith(b"P2")
    assert dumps(rgb(1, 1, [0, 0, 0])).startswith(b"P6")
    assert dumps(rgb(1, 1, [0, 0, 0]), binary=False).startswith(b"P3")


def test_comments_in_the_header_are_skipped():
    payload = b"P2\n# written by a tool that adds comments\n2 1\n255\n5 6\n"
    assert loads(payload).data == (5, 6)


def test_extra_whitespace_in_the_header_is_tolerated():
    assert loads(b"P2\n\n  2   1 \n 255 \n 5 6\n").data == (5, 6)


def test_an_unsupported_magic_is_refused():
    with pytest.raises(PnmError, match="unsupported magic"):
        loads(b"P1\n1 1\n255\n0\n")


def test_a_sixteen_bit_file_is_refused_rather_than_truncated():
    # Truncating would halve the level count, which is the quantity every
    # measurement in this repository depends on.
    with pytest.raises(PnmError, match="only 8-bit"):
        loads(b"P5\n1 1\n65535\n\x00\x00")


def test_a_truncated_binary_raster_is_refused():
    with pytest.raises(PnmError, match="truncated"):
        loads(b"P5\n4 4\n255\n\x00\x01")


def test_a_truncated_ascii_raster_is_refused():
    with pytest.raises(PnmError, match="truncated"):
        loads(b"P2\n4 4\n255\n1 2 3\n")


def test_a_header_that_ends_early_is_refused():
    with pytest.raises(PnmError, match="header ended"):
        loads(b"P5\n4 4\n")


def test_a_non_numeric_header_field_is_refused():
    with pytest.raises(PnmError, match="expected an integer"):
        loads(b"P5\nwide 4\n255\n")


def test_read_and_write_go_through_the_filesystem(tmp_path):
    image = grey(4, 4, list(range(16)))
    path = tmp_path / "image.pgm"
    write(path, image)
    assert read(path).data == image.data


def test_ascii_files_are_written_with_one_row_per_line(tmp_path):
    path = tmp_path / "image.pgm"
    write(path, grey(2, 3, [0] * 6), binary=False)
    body = path.read_text().splitlines()[3:]
    assert len(body) == 3
