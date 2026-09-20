"""An 8-bit raster, and why the bit depth is part of the argument.

Everything here stores integers in 0..255, not floats. That is deliberate and
it is the whole reason this repository exists: a dark 8-bit photograph whose
pixels all sit between 0 and 14 contains at most fifteen distinct values, and
no brightness curve applied afterwards can invent a sixteenth. A pipeline
written in floats hides that — it produces 0.0549019... and looks like it is
carrying information it does not have.

Channels are stored planar-last (r, g, b, r, g, b, ...) in a flat tuple, which
keeps the container immutable and cheap to compare in tests.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

MAX_VALUE = 255

# Rec. 601 luma weights — the ones the source report used, kept so the
# reproduction is faithful rather than improved.
LUMA_RED = 0.299
LUMA_GREEN = 0.587
LUMA_BLUE = 0.114


class RasterError(ValueError):
    """The pixel data does not describe a valid 8-bit raster."""


@dataclass(frozen=True, slots=True)
class Raster:
    """An 8-bit image. `channels` is 1 (grey) or 3 (RGB)."""

    width: int
    height: int
    channels: int
    data: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise RasterError(
                f"dimensions must be positive; got {self.width}x{self.height}"
            )
        if self.channels not in (1, 3):
            raise RasterError(f"channels must be 1 or 3; got {self.channels}")
        expected = self.width * self.height * self.channels
        if len(self.data) != expected:
            raise RasterError(
                f"expected {expected} samples for {self.width}x{self.height}"
                f"x{self.channels}; got {len(self.data)}"
            )
        for value in self.data:
            # bool is a subclass of int, and True would silently become 1.
            if isinstance(value, bool) or not isinstance(value, int):
                raise RasterError(f"samples must be int; got {type(value).__name__}")
            if not 0 <= value <= MAX_VALUE:
                raise RasterError(f"sample {value} outside 0..{MAX_VALUE}")

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.height, self.width, self.channels

    @property
    def pixel_count(self) -> int:
        return self.width * self.height

    def sample(self, y: int, x: int, channel: int = 0) -> int:
        """One sample. Out-of-range coordinates raise rather than wrap."""
        if not 0 <= y < self.height or not 0 <= x < self.width:
            raise RasterError(f"({y}, {x}) outside {self.height}x{self.width}")
        if not 0 <= channel < self.channels:
            raise RasterError(f"channel {channel} outside 0..{self.channels - 1}")
        return self.data[(y * self.width + x) * self.channels + channel]

    def clamped(self, y: int, x: int, channel: int = 0) -> int:
        """One sample with replicated edges — the border policy for filters."""
        y = max(0, min(y, self.height - 1))
        x = max(0, min(x, self.width - 1))
        return self.sample(y, x, channel)

    def plane(self, channel: int) -> tuple[int, ...]:
        """One channel as a flat tuple, row-major."""
        if not 0 <= channel < self.channels:
            raise RasterError(f"channel {channel} outside 0..{self.channels - 1}")
        return self.data[channel :: self.channels]

    def planes(self) -> tuple[tuple[int, ...], ...]:
        return tuple(self.plane(c) for c in range(self.channels))

    def luma(self) -> tuple[int, ...]:
        """Rec. 601 luma per pixel, rounded to 8-bit.

        A grey raster is its own luma, exactly — not a weighted sum of three
        copies of itself, which would round differently.
        """
        if self.channels == 1:
            return self.data
        red, green, blue = self.planes()
        return tuple(
            _quantise(LUMA_RED * r + LUMA_GREEN * g + LUMA_BLUE * b)
            for r, g, b in zip(red, green, blue, strict=True)
        )

    def to_grey(self) -> Raster:
        return Raster(self.width, self.height, 1, self.luma())

    def with_data(self, data: Iterable[int]) -> Raster:
        """A raster of the same geometry with new samples."""
        return Raster(self.width, self.height, self.channels, tuple(data))

    def map_samples(self, table: Sequence[int]) -> Raster:
        """Apply a 256-entry lookup table to every sample.

        A lookup table, not a function per pixel: every point operation in
        `enhance` is a monotone map on 0..255, and expressing them all this way
        is what makes `stats.distinct_levels` able to prove they cannot create
        detail.
        """
        if len(table) != MAX_VALUE + 1:
            raise RasterError(f"lookup table must have 256 entries; got {len(table)}")
        return self.with_data(table[value] for value in self.data)

    def render(self, levels: str = " .:-=+*#%@") -> str:
        """ASCII art of the luma, so a test failure can be read."""
        luma = self.luma()
        lines = []
        for y in range(self.height):
            row = luma[y * self.width : (y + 1) * self.width]
            lines.append(
                "".join(levels[value * (len(levels) - 1) // MAX_VALUE] for value in row)
            )
        return "\n".join(lines)


def _quantise(value: float) -> int:
    """Round to the nearest 8-bit level and clip. The only rounding rule here."""
    return max(0, min(MAX_VALUE, round(value)))


def quantise(value: float) -> int:
    return _quantise(value)


def grey(width: int, height: int, values: Iterable[float]) -> Raster:
    return Raster(width, height, 1, tuple(_quantise(v) for v in values))


def rgb(width: int, height: int, values: Iterable[float]) -> Raster:
    return Raster(width, height, 3, tuple(_quantise(v) for v in values))


def constant(width: int, height: int, value: int = 0, channels: int = 1) -> Raster:
    return Raster(width, height, channels, (value,) * (width * height * channels))
