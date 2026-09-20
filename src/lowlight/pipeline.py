"""The five-stage adaptive pipeline, reproduced faithfully, then measured.

The parameter rules below are the source report's, kept exactly — including
the `+ 0.01` stabiliser and the clamps — because a critique of a pipeline has
to run the pipeline that was described, not an improved version of it:

1. gamma 0.3 under 5% mean luminance, 0.4 under 10%, else 0.6
2. brightness = clamp(0.45 / (luma + 0.01), 1.3, 2.0),
   contrast   = clamp(0.18 / (std + 0.01),  1.2, 1.8)
3. Gaussian sigma 1.2 / 0.8 / 0.5 by measured deviation
4. sharpen strength 1.5 / 1.2 / 1.0 by measured deviation
5. percentile stretch, 1% to 99%, into 95% of the range

Each stage records what it did and what the image measured afterwards, so the
per-stage table in the README is generated rather than narrated.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from lowlight.enhance import (
    Table,
    brightness_table,
    compose,
    contrast_table,
    gamma_table,
    stretch_table,
)
from lowlight.filters import gaussian_blur, median_filter, sharpen
from lowlight.raster import MAX_VALUE, Raster
from lowlight.stats import Summary, deviation, mean, percentile, summarise

Denoiser = Callable[[Raster, float], Raster]


@dataclass(frozen=True, slots=True)
class Stage:
    """One applied stage: its name, its chosen parameter, the result."""

    name: str
    parameter: str
    summary: Summary
    table: Table | None  # None for the neighbourhood stages

    @property
    def is_point_operation(self) -> bool:
        return self.table is not None


@dataclass(frozen=True, slots=True)
class Run:
    """A complete pipeline run over one image."""

    original: Summary
    stages: tuple[Stage, ...]
    result: Raster

    @property
    def final(self) -> Summary:
        return self.stages[-1].summary

    @property
    def point_table(self) -> Table:
        """The composition of every point operation in this run.

        The four point stages collapse to this single 256-entry table. Its
        image is the set of output levels the whole chain can possibly
        produce, which is how `levels` in the final report is bounded.
        """
        tables = [stage.table for stage in self.stages if stage.table is not None]
        return compose(*tables)

    def format(self) -> str:
        lines = [
            f"{'stage':<22}{'parameter':<16}measurement",
            f"{'original':<22}{'':<16}{self.original.format()}",
        ]
        for stage in self.stages:
            lines.append(f"{stage.name:<22}{stage.parameter:<16}{stage.summary.format()}")
        return "\n".join(lines)


def choose_gamma(luma_fraction: float) -> float:
    if luma_fraction < 0.05:
        return 0.3
    if luma_fraction < 0.10:
        return 0.4
    return 0.6


def choose_brightness(luma_fraction: float) -> float:
    return min(2.0, max(1.3, 0.45 / (luma_fraction + 0.01)))


def choose_contrast(deviation_fraction: float) -> float:
    return min(1.8, max(1.2, 0.18 / (deviation_fraction + 0.01)))


def choose_sigma(deviation_fraction: float) -> float:
    if deviation_fraction > 0.25:
        return 1.2
    if deviation_fraction > 0.15:
        return 0.8
    return 0.5


def choose_sharpen(deviation_fraction: float) -> float:
    if deviation_fraction < 0.12:
        return 1.5
    if deviation_fraction < 0.18:
        return 1.2
    return 1.0


def enhance(
    raster: Raster, *, denoise: str = "gaussian", skip_stretch: bool = False
) -> Run:
    """Run the pipeline, recording every stage."""
    original = summarise(raster)
    stages: list[Stage] = []
    current = raster

    chosen_gamma = choose_gamma(original.mean_fraction)
    table = gamma_table(chosen_gamma)
    current = current.map_samples(table)
    stages.append(Stage("1 gamma", f"g={chosen_gamma:.1f}", summarise(current), table))

    luma = current.luma()
    brightness_factor = choose_brightness(mean(luma) / MAX_VALUE)
    table = brightness_table(brightness_factor)
    current = current.map_samples(table)
    stages.append(
        Stage("2a brightness", f"x{brightness_factor:.2f}", summarise(current), table)
    )

    luma = current.luma()
    contrast_factor = choose_contrast(deviation(luma) / MAX_VALUE)
    table = contrast_table(contrast_factor, pivot=mean(luma))
    current = current.map_samples(table)
    stages.append(
        Stage("2b contrast", f"x{contrast_factor:.2f}", summarise(current), table)
    )

    luma = current.luma()
    deviation_fraction = deviation(luma) / MAX_VALUE
    if denoise == "gaussian":
        sigma = choose_sigma(deviation_fraction)
        current = gaussian_blur(current, sigma)
        parameter = f"sigma={sigma:.1f}"
    elif denoise == "median":
        current = median_filter(current, 3)
        parameter = "3x3"
    elif denoise == "none":
        parameter = "skipped"
    else:
        raise ValueError(
            f"unknown denoiser {denoise!r}; expected gaussian, median or none"
        )
    stages.append(Stage(f"3 denoise {denoise}", parameter, summarise(current), None))

    strength = choose_sharpen(deviation(current.luma()) / MAX_VALUE)
    current = sharpen(current, strength)
    stages.append(Stage("4 sharpen", f"s={strength:.1f}", summarise(current), None))

    if not skip_stretch:
        luma = current.luma()
        low, high = percentile(luma, 0.01), percentile(luma, 0.99)
        table = stretch_table(low, high)
        current = current.map_samples(table)
        stages.append(Stage("5 stretch", f"{low}..{high}", summarise(current), table))

    return Run(original=original, stages=tuple(stages), result=current)
