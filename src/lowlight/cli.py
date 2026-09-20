"""`python -m lowlight` — enhance a PGM/PPM file, or run on the fixture.

With no `--input`, it uses the synthetic dark capture, so the command works in
a fresh clone with no dataset.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from lowlight import pnm, report, synthetic
from lowlight.pipeline import enhance
from lowlight.stats import summarise

DENOISERS = ("gaussian", "median", "none")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m lowlight",
        description=(
            "Low-light enhancement, measured against metrics that cannot be gamed."
        ),
    )
    parser.add_argument("--input", help="a PGM/PPM file; omit to use the fixture")
    parser.add_argument("--output", help="write the enhanced image here as PGM/PPM")
    parser.add_argument("--denoise", choices=DENOISERS, default="gaussian")
    parser.add_argument("--size", type=int, default=48, help="fixture size in pixels")
    parser.add_argument(
        "--exposure",
        type=float,
        default=0.05,
        help="fixture exposure, as a fraction of full",
    )
    parser.add_argument("--seed", type=int, default=7, help="fixture noise seed")
    parser.add_argument("--stages", action="store_true", help="print the per-stage table")
    parser.add_argument(
        "--compare",
        action="store_true",
        help="compare against a single constant multiply",
    )
    parser.add_argument(
        "--order",
        action="store_true",
        help="compare gamma-then-stretch against the reverse",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.input:
        raster = pnm.read(args.input)
        source = args.input
    else:
        pair = synthetic.dark_pair(size=args.size, exposure=args.exposure, seed=args.seed)
        raster = pair.dark
        source = (
            f"synthetic fixture {args.size}x{args.size}, "
            f"exposure {args.exposure}, seed {args.seed}"
        )

    print(f"input: {source}")
    print(f"       {summarise(raster).format()}")

    run = enhance(raster, denoise=args.denoise)
    print()
    if args.stages:
        print(run.format())
    else:
        print(f"output: {run.final.format()}")

    if args.compare:
        print()
        print(report.header())
        for comparison in report.against_gain(raster):
            print(report.row(comparison))
        print()
        print(f"levels the point stages can emit: {report.level_ceiling(run)} of 256")

    if args.order:
        print()
        print(report.header())
        for comparison in report.order_matters(raster):
            print(report.row(comparison))

    if args.output:
        pnm.write(args.output, run.result)
        print()
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
