"""Side-by-side comparisons — the part the source report is missing.

Three comparisons, each one a claim that can be checked:

* `against_gain` — the five-stage pipeline against a single multiply, scored
  on the report's own headline metric and then on metrics that cannot be
  gamed;
* `against_truth` — both against the bright original, which only exists
  because the dark image is synthesised (see `synthetic`);
* `order_matters` — gamma-then-stretch against stretch-then-gamma.
"""

from __future__ import annotations

from dataclasses import dataclass

from lowlight.enhance import Gain, apply_table, gamma_table, stretch_table
from lowlight.pipeline import Run, enhance
from lowlight.raster import MAX_VALUE, Raster
from lowlight.stats import (
    Summary,
    matched_error,
    percentile,
    relative_change,
    summarise,
)


@dataclass(frozen=True, slots=True)
class Comparison:
    """Two named results measured the same way."""

    label: str
    summary: Summary
    luminance_gain_percent: float

    @classmethod
    def build(cls, label: str, before: Summary, after: Summary) -> Comparison:
        return cls(
            label=label,
            summary=after,
            luminance_gain_percent=relative_change(before.mean, after.mean) * 100,
        )


def header() -> str:
    return (
        f"{'method':<24}{'luma +%':>10}{'levels':>8}{'entropy':>9}"
        f"{'clip hi':>9}{'std':>8}"
    )


def row(comparison: Comparison) -> str:
    summary = comparison.summary
    return (
        f"{comparison.label:<24}{comparison.luminance_gain_percent:>+10.0f}"
        f"{summary.distinct_levels:>8}{summary.entropy_bits:>9.3f}"
        f"{summary.clipped_high * 100:>8.2f}%{summary.deviation:>8.2f}"
    )


def against_gain(dark: Raster, *, factor: float | None = None) -> list[Comparison]:
    """The pipeline against a constant multiply chosen to match its brightness."""
    run = enhance(dark)
    before = run.original
    if factor is None:
        # Match the pipeline's final mean, so the comparison is at equal
        # brightness and the metric cannot prefer one for being brighter.
        factor = run.final.mean / max(before.mean, 1e-9)
    gained = Gain(factor).apply(dark)
    return [
        Comparison.build("dark original", before, before),
        Comparison.build(f"constant gain x{factor:.1f}", before, summarise(gained)),
        Comparison.build("five-stage pipeline", before, run.final),
    ]


@dataclass(frozen=True, slots=True)
class Fidelity:
    """A method's distance from the known bright original."""

    label: str
    error: float
    entropy_bits: float


def fidelity_header() -> str:
    return f"{'method':<24}{'err vs truth':>14}{'entropy':>9}"


def fidelity_row(row: Fidelity) -> str:
    return f"{row.label:<24}{row.error:>14.2f}{row.entropy_bits:>9.3f}"


def against_truth(truth: Raster, dark: Raster) -> list[Comparison]:
    """Both results against the bright original of the same scene."""
    run = enhance(dark)
    truth_summary = summarise(truth)
    return [
        Comparison.build("bright truth", truth_summary, truth_summary),
        Comparison.build("dark capture", summarise(dark), summarise(dark)),
        Comparison.build("pipeline output", run.original, run.final),
    ]


def fidelity(truth: Raster, dark: Raster) -> list[Fidelity]:
    """Brightness-matched error against the truth, for each method.

    Note what this ranks and what it does not. Entropy is printed beside it
    because the two disagree: the pipeline's blur-and-sharpen stages invent
    interpolated levels and so report *more* entropy than the ground truth
    itself, while being further from it. Entropy is invariant under point
    operations, which is why it exposes the gamma-and-gain chain — but
    neighbourhood operations inflate it, so it is not a quality metric either.
    """
    run = enhance(dark)
    truth_luma = truth.luma()
    gain_factor = run.final.mean / max(summarise(dark).mean, 1e-9)
    gained = Gain(gain_factor).apply(dark)
    return [
        Fidelity(
            "dark capture",
            matched_error(dark.luma(), truth_luma),
            summarise(dark).entropy_bits,
        ),
        Fidelity(
            f"constant gain x{gain_factor:.1f}",
            matched_error(gained.luma(), truth_luma),
            summarise(gained).entropy_bits,
        ),
        Fidelity(
            "five-stage pipeline",
            matched_error(run.result.luma(), truth_luma),
            run.final.entropy_bits,
        ),
    ]


def order_matters(dark: Raster, *, gamma: float = 0.4) -> list[Comparison]:
    """Gamma then stretch, against stretch then gamma. Same two operations."""
    before = summarise(dark)

    first = apply_table(dark, gamma_table(gamma))
    luma = first.luma()
    first = apply_table(
        first, stretch_table(percentile(luma, 0.01), percentile(luma, 0.99))
    )

    luma = dark.luma()
    second = apply_table(
        dark, stretch_table(percentile(luma, 0.01), percentile(luma, 0.99))
    )
    second = apply_table(second, gamma_table(gamma))

    return [
        Comparison.build("gamma then stretch", before, summarise(first)),
        Comparison.build("stretch then gamma", before, summarise(second)),
    ]


def level_ceiling(run: Run) -> int:
    """How many distinct output levels the run's point operations can produce.

    The composed table's image size. Anything the final image shows beyond
    this came from the two neighbourhood stages — blurring and sharpening,
    which average neighbours and so *can* create intermediate levels. Which
    also means those levels are interpolation, not recovered detail.
    """
    return len(set(run.point_table))


def full_scale_levels() -> int:
    return MAX_VALUE + 1
