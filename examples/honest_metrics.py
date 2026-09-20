"""What "+664% luminance" is worth, measured five ways.

No dataset, no network, no dependencies:

    python examples/honest_metrics.py
"""

from __future__ import annotations

from lowlight import report, synthetic
from lowlight.filters import gaussian_blur, noise_estimate, sharpen
from lowlight.pipeline import enhance
from lowlight.stats import deviation, matched_error, summarise

RULE = "=" * 78
SIZE = 48


def section(number: int, title: str) -> None:
    print()
    print(RULE)
    print(f"{number}. {title}")
    print()


def main() -> None:
    pair = synthetic.dark_pair(size=SIZE)
    print(
        f"a {SIZE}x{SIZE} test pattern at {pair.exposure:.0%} exposure, seed {pair.seed}"
    )
    print(f"truth: {summarise(pair.truth).format()}")
    print(f"dark:  {summarise(pair.dark).format()}")

    section(1, "The pipeline, stage by stage")
    run = enhance(pair.dark)
    print(run.format())
    print()
    print("  Look at the 'levels' and 'entropy' columns for the first three")
    print("  stages. Mean luminance rises from 2.84% to 44.44% — the report's")
    print("  '+1465%' — and the level count does not move at all, nor does the")
    print("  entropy, to three decimal places. Those three stages are lookup")
    print("  tables. A lookup table maps 256 levels onto 256 levels: it can")
    print("  merge two levels into one, never split one into two.")
    print()
    print(
        f"  All four point stages composed can emit "
        f"{report.level_ceiling(run)} distinct levels."
    )
    print(
        f"  The final image shows {run.final.distinct_levels}. Every extra level came "
        f"from the"
    )
    print("  blur and the sharpen averaging neighbours together — interpolation")
    print("  between samples, not detail recovered from the sensor.")

    section(2, "A single multiply scores the same on the reported metric")
    print(report.header())
    for comparison in report.against_gain(pair.dark):
        print(report.row(comparison))
    print()
    print("  One multiplication, no adaptive anything, and 'luminance")
    print("  improvement' cannot separate it from five stages of processing.")
    print("  It does clip 12.6% of the image to pure white, which the source")
    print("  report's metrics do not charge for, and which is irreversible.")

    section(3, "Standard deviation is not contrast")
    field = synthetic.noise_field(32)
    before = deviation(field.luma())
    after = deviation(sharpen(field, 1.2).luma())
    print(f"  a field of pure noise, no structure at all: std {before:.2f}")
    print(f"  the same field, sharpened:                    std {after:.2f}")
    print(
        f"  '+{(after / before - 1) * 100:.0f}% contrast' on an image containing "
        f"no information."
    )
    print()
    blurred = gaussian_blur(field, 0.8)
    resharpened = sharpen(blurred, 1.2)
    print(f"  noise estimate, raw field:        {noise_estimate(field):.4f}")
    print(f"  after the denoise stage:          {noise_estimate(blurred):.4f}")
    print(f"  after the sharpen stage that follows it: {noise_estimate(resharpened):.4f}")
    recovered = noise_estimate(resharpened) / noise_estimate(field)
    print(f"  Stage 4 puts back {recovered:.0%} of the noise stage 3 removed.")

    section(4, "Measured against the truth, the pipeline is further away")
    print(report.fidelity_header())
    for row in report.fidelity(pair.truth, pair.dark):
        print(report.fidelity_row(row))
    print()
    print(
        f"  The bright original's own entropy is "
        f"{summarise(pair.truth).entropy_bits:.3f} bits — *less* than the"
    )
    print("  pipeline output's. So entropy is not a quality metric either: it")
    print("  resists point operations, which is what exposes stage 1 to 3, but")
    print("  neighbourhood operations inflate it by inventing levels.")
    print("  Error against the known original is the only column here that")
    print("  cannot be improved by doing more to the image — and it exists only")
    print("  because the fixture is synthetic. On a real night photograph there")
    print("  is no truth to subtract from.")

    section(5, "The pipeline is right, in a regime its report never identified")
    print(
        f"{'noise':>7}{'do nothing':>12}{'gain':>9}{'pipeline':>10}{'median':>9}   best"
    )
    for noise in (0.0, 1.5, 4.0, 8.0, 15.0):
        sample = synthetic.dark_pair(size=SIZE, noise=noise)
        truth_luma = sample.truth.luma()
        rows = report.fidelity(sample.truth, sample.dark)
        median_error = matched_error(
            enhance(sample.dark, denoise="median").result.luma(), truth_luma
        )
        scores = {
            "nothing": rows[0].error,
            "gain": rows[1].error,
            "pipeline": rows[2].error,
            "median": median_error,
        }
        best = min(scores, key=scores.get)
        print(
            f"{noise:>7.1f}{rows[0].error:>12.2f}{rows[1].error:>9.2f}"
            f"{rows[2].error:>10.2f}{median_error:>9.2f}   {best}"
        )
    print()
    print("  Below noise 4 the pipeline is worse than leaving the image alone:")
    print("  it destroys more than it recovers. Above it the pipeline wins by a")
    print("  wide margin, and the whole benefit comes from the denoise stage.")
    print("  So the pipeline is worth running — in one regime. Its own reported")
    print("  metrics improve in both, which is why they could not locate the")
    print("  boundary, and why the original conclusion ('elementary techniques")
    print("  are highly effective') is true by accident.")


if __name__ == "__main__":
    main()
