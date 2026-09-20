"""Neighbourhood filters: Gaussian, median, and unsharp masking.

The decision this module exists to make explicit is Gaussian against median
for noise. The usual summary — "median preserves edges" — is true and
incomplete. A median filter also *deletes* any feature thinner than half its
window, unconditionally and without trace: a one-pixel bright line in a 3x3
median is gone, because it is never the middle value of nine samples. On a
photograph that removes sensor hot pixels; on an image with real thin
structure it removes the structure.

Borders replicate, via `Raster.clamped`, so no filter darkens the edge the way
zero padding would.
"""

from __future__ import annotations

import math

from lowlight.raster import MAX_VALUE, Raster, RasterError, quantise

SHARPEN_BASE: tuple[tuple[float, ...], ...] = (
    (0.0, -1.0, 0.0),
    (-1.0, 5.0, -1.0),
    (0.0, -1.0, 0.0),
)


def gaussian_kernel(sigma: float) -> list[float]:
    """A normalised 1-D Gaussian, radius 3 sigma.

    Separable: a 2-D Gaussian blur is two 1-D passes, which turns an O(r^2)
    filter into O(r) per pixel. That is not a micro-optimisation at sigma=1.2
    in pure Python — it is the difference between the demo running and not.
    """
    if sigma <= 0:
        raise RasterError(f"sigma must be positive; got {sigma}")
    radius = max(1, math.ceil(3 * sigma))
    weights = [
        math.exp(-(offset**2) / (2 * sigma * sigma))
        for offset in range(-radius, radius + 1)
    ]
    total = sum(weights)
    return [w / total for w in weights]


def gaussian_blur(raster: Raster, sigma: float) -> Raster:
    """Separable Gaussian blur, applied per channel."""
    kernel = gaussian_kernel(sigma)
    radius = len(kernel) // 2

    horizontal = [0.0] * len(raster.data)
    for y in range(raster.height):
        for x in range(raster.width):
            for channel in range(raster.channels):
                total = 0.0
                for offset, weight in enumerate(kernel, start=-radius):
                    total += weight * raster.clamped(y, x + offset, channel)
                horizontal[(y * raster.width + x) * raster.channels + channel] = total

    def read(y: int, x: int, channel: int) -> float:
        y = max(0, min(y, raster.height - 1))
        x = max(0, min(x, raster.width - 1))
        return horizontal[(y * raster.width + x) * raster.channels + channel]

    output = []
    for y in range(raster.height):
        for x in range(raster.width):
            for channel in range(raster.channels):
                total = 0.0
                for offset, weight in enumerate(kernel, start=-radius):
                    total += weight * read(y + offset, x, channel)
                output.append(quantise(total))
    return raster.with_data(output)


def median_filter(raster: Raster, size: int = 3) -> Raster:
    """Window median, per channel. Preserves edges; deletes thin features."""
    if size <= 0 or size % 2 == 0:
        raise RasterError(f"median window must be positive and odd; got {size}")
    half = size // 2
    output = []
    for y in range(raster.height):
        for x in range(raster.width):
            for channel in range(raster.channels):
                window = [
                    raster.clamped(y + dy, x + dx, channel)
                    for dy in range(-half, half + 1)
                    for dx in range(-half, half + 1)
                ]
                window.sort()
                output.append(window[len(window) // 2])
    return raster.with_data(output)


def sharpen(raster: Raster, strength: float = 1.0) -> Raster:
    """Unsharp masking through a 3x3 kernel whose weights sum to 1.

    The centre weight is 4 + strength and the four neighbours are -1 each, so
    the kernel sums to `strength`... which is why it is divided by `strength`
    afterwards. Skipping that division is the standard bug: it brightens the
    image proportionally to the sharpening amount, and then the brightness gain
    gets reported as a sharpening benefit.
    """
    if strength <= 0:
        raise RasterError(f"strength must be positive; got {strength}")
    kernel = [list(row) for row in SHARPEN_BASE]
    kernel[1][1] = 4.0 + strength
    total = sum(sum(row) for row in kernel)

    output = []
    for y in range(raster.height):
        for x in range(raster.width):
            for channel in range(raster.channels):
                accumulated = 0.0
                for ky in range(3):
                    for kx in range(3):
                        weight = kernel[ky][kx]
                        if weight:
                            accumulated += weight * raster.clamped(
                                y + ky - 1, x + kx - 1, channel
                            )
                output.append(quantise(accumulated / total))
    return raster.with_data(output)


def noise_estimate(raster: Raster) -> float:
    """A crude noise estimate: mean absolute Laplacian.

    Crude on purpose — it is used only to compare the same image before and
    after a filter, where the bias cancels. Reporting it as an absolute noise
    level would be a different and unsupported claim.
    """
    luma = raster.to_grey()
    total = 0.0
    for y in range(luma.height):
        for x in range(luma.width):
            centre = 4 * luma.clamped(y, x)
            surround = (
                luma.clamped(y - 1, x)
                + luma.clamped(y + 1, x)
                + luma.clamped(y, x - 1)
                + luma.clamped(y, x + 1)
            )
            total += abs(centre - surround)
    return total / (luma.pixel_count * MAX_VALUE)
