"""Point operations, every one of them a 256-entry lookup table.

Writing gamma, brightness, contrast and histogram stretching as lookup tables
is not a performance trick. It is the proof: a lookup table is a function from
256 levels to 256 levels, so it can map two input levels onto one output level
but never one onto two. No sequence of these operations, in any order, with
any parameters, can increase the number of distinct levels in the image.

That single fact is what the source report's "+664% luminance" is hiding.
"""

from __future__ import annotations

from dataclasses import dataclass

from lowlight.raster import MAX_VALUE, Raster, RasterError, quantise

Table = tuple[int, ...]

IDENTITY: Table = tuple(range(MAX_VALUE + 1))


def gamma_table(gamma: float) -> Table:
    """`out = (in/255) ** gamma * 255`. Gamma below 1 lifts the shadows."""
    if gamma <= 0:
        raise RasterError(f"gamma must be positive; got {gamma}")
    return tuple(
        quantise(((value / MAX_VALUE) ** gamma) * MAX_VALUE)
        for value in range(MAX_VALUE + 1)
    )


def brightness_table(factor: float) -> Table:
    """A multiply, as PIL's ImageEnhance.Brightness does it."""
    if factor < 0:
        raise RasterError(f"brightness factor must be non-negative; got {factor}")
    return tuple(quantise(value * factor) for value in range(MAX_VALUE + 1))


def contrast_table(factor: float, *, pivot: float = 128.0) -> Table:
    """Scale distance from a pivot. PIL pivots on the image mean; this pivots
    on mid-grey by default, and `pipeline` passes the measured mean so the
    reproduction matches."""
    if factor < 0:
        raise RasterError(f"contrast factor must be non-negative; got {factor}")
    return tuple(
        quantise(pivot + (value - pivot) * factor) for value in range(MAX_VALUE + 1)
    )


def stretch_table(low: int, high: int, *, headroom: float = 0.95) -> Table:
    """Map [low, high] onto [0, 255*headroom], clipping outside.

    `headroom` reproduces the source's "95% of the range" choice. Note what it
    costs: everything below `low` becomes 0 and everything above `high` becomes
    the ceiling, so a 1%/99% stretch deliberately clips 2% of the image. That
    is a defensible trade and an undisclosed one in the original.
    """
    if not 0 <= low <= MAX_VALUE or not 0 <= high <= MAX_VALUE:
        raise RasterError(f"bounds must be in 0..255; got {low}..{high}")
    if high <= low:
        # A flat image has low == high. Returning identity keeps the pipeline
        # running instead of dividing by zero.
        return IDENTITY
    ceiling = MAX_VALUE * headroom
    span = high - low
    return tuple(
        quantise(min(ceiling, max(0.0, (value - low) / span * ceiling)))
        for value in range(MAX_VALUE + 1)
    )


def compose(*tables: Table) -> Table:
    """One table equivalent to applying each in turn, left to right.

    Composition is the honest way to show that a chain of point operations is
    itself a point operation — and therefore that the whole chain, however
    long, still cannot add a level.
    """
    result = IDENTITY
    for table in tables:
        if len(table) != MAX_VALUE + 1:
            raise RasterError(f"lookup table must have 256 entries; got {len(table)}")
        result = tuple(table[result[value]] for value in range(MAX_VALUE + 1))
    return result


def apply_table(raster: Raster, table: Table) -> Raster:
    return raster.map_samples(table)


def gamma(raster: Raster, value: float) -> Raster:
    return apply_table(raster, gamma_table(value))


def brightness(raster: Raster, factor: float) -> Raster:
    return apply_table(raster, brightness_table(factor))


def contrast(raster: Raster, factor: float, *, pivot: float = 128.0) -> Raster:
    return apply_table(raster, contrast_table(factor, pivot=pivot))


@dataclass(frozen=True, slots=True)
class Gain:
    """A constant multiply — the baseline the report's metric prefers.

    This exists to be compared against the five-stage pipeline. It is one
    multiplication, it adds no information whatsoever, and on "mean luminance
    improvement" it wins. Any metric that ranks this first is not measuring
    image quality.
    """

    factor: float

    def apply(self, raster: Raster) -> Raster:
        return brightness(raster, self.factor)
