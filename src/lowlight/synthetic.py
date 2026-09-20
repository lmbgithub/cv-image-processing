"""A dark image with known ground truth, so the measurements can be checked.

The real dataset (see the README) is 6,000 night photographs and is not
committed here. Everything in this repository is therefore verified against a
generated scene whose properties are known exactly:

* a bright test pattern is built at full range,
* it is then *crushed*: scaled down to a few percent of full scale and
  requantised to 8 bits, which is what a short exposure physically does,
* optional shot noise is added with a recorded seed.

Because the bright original is available, "how much did the pipeline recover?"
has an answer here that it does not have on a real photograph: compare against
the truth, not against how the result looks.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from lowlight.raster import MAX_VALUE, Raster, RasterError, grey, quantise


@dataclass(frozen=True, slots=True)
class DarkPair:
    """The bright truth and the dark capture of the same scene."""

    truth: Raster
    dark: Raster
    exposure: float
    seed: int | None


def pattern(size: int = 48) -> Raster:
    """Gradients, bars, a disc and a one-pixel line, at full range.

    The one-pixel line is there to be destroyed by the median filter, and the
    smooth gradient is there to show banding once the image is crushed.
    """
    if size < 16:
        raise RasterError(f"size must be at least 16; got {size}")
    values = []
    for y in range(size):
        for x in range(size):
            if y == size // 2:
                value = MAX_VALUE  # one-pixel bright line
            elif (y - size // 4) ** 2 + (x - size // 4) ** 2 <= (size // 6) ** 2:
                value = 200  # a disc
            elif y > size * 3 // 4:
                value = MAX_VALUE if (x // 3) % 2 else 20  # bars
            else:
                value = x * MAX_VALUE // (size - 1)  # horizontal gradient
            values.append(value)
    return grey(size, size, values)


def crush(
    raster: Raster, exposure: float, *, seed: int | None = None, noise: float = 0.0
) -> Raster:
    """Simulate a short exposure: scale down, add noise, requantise.

    The requantisation is the destructive step and it happens once, here, in
    the same order the sensor does it. Scaling a float image down and back up
    losslessly is the mistake that makes a simulated low-light experiment
    easier than the real one.
    """
    if not 0.0 < exposure <= 1.0:
        raise RasterError(f"exposure must be in (0, 1]; got {exposure}")
    if noise < 0.0:
        raise RasterError(f"noise must be non-negative; got {noise}")
    generator = random.Random(seed) if noise else None
    values = []
    for value in raster.data:
        scaled = value * exposure
        if generator is not None:
            scaled += generator.gauss(0.0, noise)
        values.append(quantise(scaled))
    return raster.with_data(values)


def dark_pair(
    *, size: int = 48, exposure: float = 0.05, noise: float = 1.5, seed: int = 7
) -> DarkPair:
    """The standard fixture: a 5%-exposure capture of the test pattern."""
    truth = pattern(size)
    return DarkPair(
        truth=truth,
        dark=crush(truth, exposure, seed=seed, noise=noise),
        exposure=exposure,
        seed=seed,
    )


def noise_field(size: int = 32, *, level: int = 8, seed: int = 11) -> Raster:
    """Pure noise around a dim mean — no structure at all.

    Used to show that standard deviation rises when this is sharpened. A
    "contrast" metric that improves on an image containing no information is
    not measuring contrast.
    """
    generator = random.Random(seed)
    return grey(
        size, size, (generator.gauss(level, level / 3) for _ in range(size * size))
    )
