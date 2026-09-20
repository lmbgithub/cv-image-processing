"""Measurements, including the ones that make the headline number collapse.

The source report's summary is "luminance +664%, contrast +460%". Both are
real numbers and neither is evidence of an improved image:

* **mean luminance** is maximised by multiplying every pixel by a constant,
  which adds nothing;
* **standard deviation** is increased by sharpening, and sharpening a pure
  noise field increases it most of all.

So this module computes those two, because the reproduction has to be
faithful, and next to them the three that resist being gamed: the number of
distinct levels actually present, the Shannon entropy of the histogram, and
the fraction of samples pinned at 0 or 255. A point operation on an 8-bit
image cannot increase the first two, and the third only ever goes up.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from lowlight.raster import MAX_VALUE, Raster


@dataclass(frozen=True, slots=True)
class Summary:
    """Everything worth printing about one image, measured on its luma."""

    mean: float
    deviation: float
    minimum: int
    maximum: int
    distinct_levels: int
    entropy_bits: float
    clipped_low: float
    clipped_high: float
    percentile_1: int
    percentile_99: int

    @property
    def mean_fraction(self) -> float:
        """Mean as a fraction of full scale — the report's '5.5%' figure."""
        return self.mean / MAX_VALUE

    @property
    def dynamic_range(self) -> int:
        return self.maximum - self.minimum

    def format(self) -> str:
        return (
            f"mean {self.mean:6.2f} ({self.mean_fraction * 100:5.2f}%)  "
            f"std {self.deviation:6.2f}  range {self.minimum:3d}..{self.maximum:3d}  "
            f"levels {self.distinct_levels:3d}  entropy {self.entropy_bits:5.3f} bits  "
            f"clipped {self.clipped_low * 100:5.2f}%/{self.clipped_high * 100:5.2f}%"
        )


def histogram(values: Sequence[int]) -> list[int]:
    """A 256-bin count. Bins are levels, not ranges — no binning choice to make."""
    counts = [0] * (MAX_VALUE + 1)
    for value in values:
        counts[value] += 1
    return counts


def mean(values: Sequence[int]) -> float:
    if not values:
        raise ValueError("cannot take the mean of no samples")
    return sum(values) / len(values)


def deviation(values: Sequence[int]) -> float:
    """Population standard deviation."""
    if not values:
        raise ValueError("cannot take the deviation of no samples")
    average = mean(values)
    return math.sqrt(sum((v - average) ** 2 for v in values) / len(values))


def percentile(values: Sequence[int], fraction: float) -> int:
    """Nearest-rank percentile on the sorted samples.

    Nearest-rank, not interpolated: the samples are integer levels, and an
    interpolated percentile would return 12.4 — a level that is not in the
    image and cannot be mapped to.
    """
    if not values:
        raise ValueError("cannot take a percentile of no samples")
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"fraction must be in [0, 1]; got {fraction}")
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return ordered[index]


def distinct_levels(values: Sequence[int]) -> int:
    """How many of the 256 levels actually occur.

    This is the number the report never printed. A photograph whose samples
    span 0..14 has at most 15 distinct levels, and every point operation that
    follows is a function from 0..255 to 0..255 — so it can merge levels but
    never separate them. Brightening a dark image rescales a fifteen-level
    image into a fifteen-level image.
    """
    return len(set(values))


def entropy_bits(values: Sequence[int]) -> float:
    """Shannon entropy of the level histogram, in bits per sample."""
    if not values:
        raise ValueError("cannot take the entropy of no samples")
    total = len(values)
    result = 0.0
    for count in histogram(values):
        if count:
            probability = count / total
            result -= probability * math.log2(probability)
    return result


def clipped_fractions(values: Sequence[int]) -> tuple[float, float]:
    """Fraction of samples sitting exactly at 0 and exactly at 255.

    Both are irreversible: once a sample is pinned to the rail, the operation
    that put it there cannot be undone and the detail it carried is gone. The
    source pipeline reports neither.
    """
    if not values:
        raise ValueError("cannot measure clipping on no samples")
    low = sum(1 for v in values if v == 0)
    high = sum(1 for v in values if v == MAX_VALUE)
    return low / len(values), high / len(values)


def summarise(raster: Raster) -> Summary:
    """Every measurement above, on the raster's luma."""
    luma = raster.luma()
    low, high = clipped_fractions(luma)
    return Summary(
        mean=mean(luma),
        deviation=deviation(luma),
        minimum=min(luma),
        maximum=max(luma),
        distinct_levels=distinct_levels(luma),
        entropy_bits=entropy_bits(luma),
        clipped_low=low,
        clipped_high=high,
        percentile_1=percentile(luma, 0.01),
        percentile_99=percentile(luma, 0.99),
    )


def relative_change(before: float, after: float) -> float:
    """The report's "+664%" figure: (after - before) / before.

    Kept, with the stabiliser the source used, so the headline number can be
    reproduced exactly — and then shown to be meaningless. On a mean of 0.05
    the denominator is tiny, which is the only reason the percentage is large.
    """
    return (after - before) / (before + 1e-9)


def mean_absolute_error(left: Sequence[int], right: Sequence[int]) -> float:
    """Mean |difference| between two equally sized sample sequences."""
    if len(left) != len(right):
        raise ValueError(f"{len(left)} samples against {len(right)}")
    if not left:
        raise ValueError("cannot compare no samples")
    return sum(abs(a - b) for a, b in zip(left, right, strict=True)) / len(left)


def matched_error(result: Sequence[int], truth: Sequence[int]) -> float:
    """Mean absolute error after scaling `result` to `truth`'s mean.

    The fidelity metric, and the only one here that cannot be improved by
    doing more to the image. Brightness is divided out first because every
    method under comparison changes it deliberately — without that, the
    metric would just re-measure exposure and rank the brightest result last.

    This number exists only because the bright original exists, which is the
    argument for a synthetic fixture: on a real night photograph there is no
    truth to subtract, and "it looks better" is the only available claim.
    """
    if not truth:
        raise ValueError("cannot compare no samples")
    truth_mean = mean(truth)
    result_mean = mean(result)
    if result_mean == 0:
        return truth_mean
    scale = truth_mean / result_mean
    scaled = [max(0, min(MAX_VALUE, round(v * scale))) for v in result]
    return mean_absolute_error(scaled, truth)
