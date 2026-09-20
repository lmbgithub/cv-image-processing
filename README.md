# cv-image-processing

A faithful reproduction of an adaptive five-stage low-light enhancement pipeline, together with the measurements that show what it actually recovers.

The source coursework (kept in `legacy/task.txt`) reports "luminance improved 664% to 1493%, contrast improved 351% to 871%" and concludes that elementary techniques are highly effective. Both percentages are arithmetically correct, and neither is evidence. The project's intention is to show why, with numbers instead of argument:

- a **single multiplication** matches the five-stage pipeline on the reported luminance metric;
- the first three stages raise mean luminance by **1465%** while the image's entropy stays at **3.945 bits — unchanged to three decimals**;
- **sharpening pure noise** raises "contrast" by **258%**;

**Standard library only — no numpy, no scipy, no Pillow. 188 tests.**

## Skills demonstrated

**Image processing** — gamma, brightness, contrast and percentile histogram stretching as composable 256-entry lookup tables; separable Gaussian blur; median filtering; unsharp masking; 8-bit quantisation and clipping analysis

**Measurement rigour** — distinguishing metrics that can be gamed from metrics that cannot; Shannon entropy and distinct-level counting as information measures; brightness-matched mean absolute error against ground truth; regime analysis across a noise sweep to locate where a method actually helps

**Signal / information theory** — proving that any chain of point operations is itself a point operation, and therefore cannot increase the level count or the entropy of an 8-bit image

**Software engineering** — pure standard library including a from-scratch Netpbm (PGM/PPM) codec, 188 tests covering malformed input and degenerate cases, deterministic seeded fixtures, argparse CLI, CI across Python 3.10-3.12

**Architecture** — `stats` imports nothing from `enhance` or `pipeline`, so the measurements cannot be tuned to flatter the pipeline they judge

## What the run looks like

```
$ python examples/honest_metrics.py
a 48x48 test pattern at 5% exposure, seed 7
truth: mean 143.67 (56.34%)  std  87.04  range   0..255  levels  49  entropy 5.000 bits  clipped  1.56%/15.10%
dark:  mean   7.24 ( 2.84%)  std   4.48  range   0.. 17  levels  18  entropy 3.945 bits  clipped  9.11%/ 0.00%

==============================================================================
1. The pipeline, stage by stage

stage                 parameter       measurement
original                              mean   7.24 ( 2.84%)  std   4.48  range   0.. 17  levels  18  entropy 3.945 bits  clipped  9.11%/ 0.00%
1 gamma               g=0.3           mean  78.77 (30.89%)  std  29.83  range   0..113  levels  18  entropy 3.945 bits  clipped  9.11%/ 0.00%
2a brightness         x1.41           mean 111.31 (43.65%)  std  42.12  range   0..159  levels  18  entropy 3.945 bits  clipped  9.11%/ 0.00%
2b contrast           x1.20           mean 113.33 (44.44%)  std  45.34  range   0..169  levels  18  entropy 3.945 bits  clipped  9.11%/ 0.00%
3 denoise gaussian    sigma=0.8       mean 113.33 (44.44%)  std  35.59  range   7..158  levels 149  entropy 6.718 bits  clipped  0.00%/ 0.00%
4 sharpen             s=1.2           mean 113.64 (44.56%)  std  46.23  range   0..206  levels 199  entropy 7.008 bits  clipped  2.69%/ 0.00%
5 stretch             0..187          mean 147.14 (57.70%)  std  59.77  range   0..242  levels 186  entropy 6.970 bits  clipped  2.69%/ 0.00%

  All four point stages composed can emit 24 distinct levels.
  The final image shows 186.

==============================================================================
2. A single multiply scores the same on the reported metric

method                     luma +%  levels  entropy  clip hi     std
dark original                   +0      18    3.945    0.00%    4.48
constant gain x20.3          +1892      14    3.744   12.59%   86.89
five-stage pipeline          +1932     186    6.970    0.00%   59.77

==============================================================================
3. Standard deviation is not contrast

  a field of pure noise, no structure at all: std 2.71
  the same field, sharpened:                    std 9.69
  '+258% contrast' on an image containing no information.

  noise estimate, raw field:        0.0365
  after the denoise stage:          0.0065
  after the sharpen stage that follows it: 0.0317
  Stage 4 puts back 87% of the noise stage 3 removed.

==============================================================================
4. Measured against the truth, the pipeline is further away

method                    err vs truth  entropy
dark capture                     20.81    3.945
constant gain x20.3              20.75    3.744
five-stage pipeline              35.41    6.970

==============================================================================
5. The pipeline is right, in a regime its report never identified

  noise  do nothing     gain  pipeline   median   best
    0.0        3.25     5.27     34.93    35.10   nothing
    1.5       20.81    20.75     35.41    41.09   gain
    4.0       47.98    48.00     42.77    57.73   pipeline
    8.0       75.29    75.57     46.42    75.80   pipeline
   15.0       95.64    96.30     62.66    88.39   pipeline
```

Every number is real output from a seeded run. Reproduce with `python examples/honest_metrics.py`.

## The six decisions worth discussing

### 1. The pixels are 8-bit integers, not floats

This is the decision the rest of the repository rests on. A night photograph whose samples all fall between 0 and 17 holds **eighteen distinct values**, and that is the whole information content of the file. Writing the pipeline in floats produces `0.0549019607…` and hides it: the arithmetic looks like it is carrying precision the sensor never captured.

Keeping everything in 0..255 makes the ceiling visible and countable, which is what turns the critique from a rhetorical point into `levels 18` printed in a column.

### 2. Every point operation is a 256-entry lookup table

Gamma, brightness, contrast and histogram stretching are all implemented as `tuple[int, ...]` of length 256, applied with `Raster.map_samples`. Not for speed — for proof.

A lookup table is a function from 256 levels to 256 levels. It can map two input levels onto one output level; it cannot map one onto two. So **no sequence of these operations, in any order, with any parameters, can increase the number of distinct levels in the image.** `enhance.compose` collapses a whole chain into one table and `test_no_chain_of_point_operations_can_increase_the_level_count` asserts it.

That is why stages 1, 2a and 2b show `levels 18, entropy 3.945` three times in a row while the reported luminance climbs 1465%. The "+664%" in the original is a relabelling of the same eighteen values.

### 3. The level count rises only at the blur step

`levels` jumps from 18 to 149 at stage 3. New levels are arithmetically possible there, because averaging neighbours produces values between the samples. They are interpolation between eighteen real measurements, not detail recovered from the sensor.

This is measurable: the four point stages composed can emit **24** levels, and the final image shows **186**. The difference is manufactured by the two neighbourhood stages. A reviewer who reads "the histogram now uses the full dynamic range" as "the image now contains more information" has the causality backwards.

### 4. Three metrics, and three different ways to game them

| metric | immune to | inflated by |
|---|---|---|
| mean luminance | nothing | a single multiply |
| standard deviation | nothing | sharpening, including pure noise (+258%) |
| entropy / level count | every point operation | blurring and sharpening |

The third row is the one that catches people out. Entropy is invariant under gamma and gain, which is exactly what exposes stages 1 to 3 — which invites the assumption that it measures quality. It does not. The pipeline's output reports **6.970 bits against the ground truth's 5.000**, because blur and sharpen invent interpolated levels, while sitting further from the truth than doing nothing at all. Entropy is a diagnostic for point operations, not a quality score.

The only column in section 4 that cannot be improved by doing more to the image is error against the known original — and it exists only because the fixture is synthetic.

### 5. The fixture is synthetic, which is the point

`synthetic.crush` takes a bright test pattern and simulates a short exposure: scale down, add seeded noise, **requantise to 8 bits once**, in that order — the order a sensor does it. Scaling a float image down and back up losslessly is the mistake that makes a simulated low-light experiment easier than the real one.

Because the bright original is retained, "how much did the pipeline recover?" has an answer here that it does not have on a real photograph. On the DarkFace images there is no truth to subtract from, so the only available claim is that the output looks better — which is how a pipeline that moves *away* from the truth at low noise gets published as a success.

Brightness is divided out before the error is taken, because every method under comparison changes exposure deliberately; without that the metric would just re-measure brightness.

### 6. Median against Gaussian, stated completely

"The median filter preserves edges" is true and incomplete. A median filter also **deletes any feature thinner than half its window**, unconditionally: a one-pixel bright line is never the middle value of nine samples, so it is gone without a trace. In the fixture, that line goes from 255 to 22 under a 3x3 median, while a Gaussian at sigma 0.8 keeps it at 133.

On a photograph, that removes sensor hot pixels. On an image with real thin structure it removes the structure — and the median column in section 5 is worse than Gaussian at every noise level tested here for exactly that reason.

### And the crossover

Section 5 is the honest verdict. Below noise 4 the pipeline is **worse than leaving the image alone** (34.93 against 3.25 at zero noise): it destroys more than it recovers. Above it the pipeline wins by a wide margin (46.42 against 75.29 at noise 8), and essentially all of the benefit comes from the denoise stage — the one stage that is not a lookup table.

So the original conclusion is defensible in the regime where the images actually are noisy, and its reported metrics improved in both regimes, which is why they could not locate the boundary. Being right for a reason you have not measured is not the same as being right.

## Design

```
raster.py     8-bit samples and geometry. Knows nothing about enhancement.
pnm.py        PGM/PPM read and write, so a real photograph needs no dependency
stats.py      measurements only — never modifies an image
enhance.py    point operations, each a 256-entry lookup table
filters.py    Gaussian, median, unsharp mask, and a noise estimate
synthetic.py  a dark capture whose bright original is known
pipeline.py   the source report's five adaptive stages, reproduced exactly
report.py     the comparisons the source report is missing
cli.py        python -m lowlight
```

The load-bearing boundary is that `stats` imports nothing from `enhance` or `pipeline`. The measurements cannot be tuned to flatter the pipeline because they do not know it exists.

The parameter rules in `pipeline.py` — including the `+ 0.01` stabiliser and every clamp — are the source's, and `test_pipeline.py` pins them against the values quoted in the original report (gamma 0.4 and brightness 1.80x for the worked example). A critique has to run the pipeline that was described.

Known truth, checked rather than assumed: `pattern` has exact geometry, `test_luma_uses_rec_601_weights` checks the luma of pure red against 76, and the PGM round trip is asserted byte for byte. A measurement tool never checked against known truth is measuring its own bugs.

## Usage

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

pytest                                   # 188 tests
python examples/honest_metrics.py        # no dataset, no network

python -m lowlight --stages --compare --order
python -m lowlight --denoise median --stages
python -m lowlight --input dark.pgm --output bright.pgm
```

As a library:

```python
from lowlight.pipeline import enhance
from lowlight.synthetic import dark_pair

run = enhance(dark_pair().dark)
print(run.format())
print(run.final.distinct_levels, "levels from", len(set(run.point_table)), "reachable")
```

## Dataset

**None required.** Every number above comes from `synthetic.py`.

To run on the images the original report used, the **DarkFace** dataset (6,000 night photographs, Yang et al., *Advancing Image Understanding in Poor Visibility Environments*, IEEE TIP 29:5737-5752, 2020):

1. Request access and download from the benchmark page: <https://flyywh.github.io/CVPRW2019LowLight/>
2. The archive holds JPEGs; this package reads Netpbm. Convert the four images the report used with any of:

```bash
# ImageMagick
magick 1002.png -colorspace Gray -depth 8 -resize 512x512 1002.pgm

# Netpbm
pngtopnm 1002.png | pnmtopgm > 1002.pgm

# macOS, no install
sips -s format png 1002.png --out tmp.png && magick tmp.png 1002.ppm
```

3. Run it:

```bash
python -m lowlight --input 1002.pgm --stages --compare
```

Images and datasets are **not committed** to this repository, and `.gitignore` excludes `*.png`, `*.jpg`, `*.pgm` and `*.ppm` so they cannot be added by accident.

## Scope

- **Retinex, CLAHE, BM3D, or any learned enhancer.** All of them would score better. None of them would change the argument, which is about what the reported metrics measure, and adding a method whose mechanism is not written out here would weaken the one claim this repository can fully defend.
- **A perceptual quality metric (SSIM, LPIPS).** SSIM needs a reference image, so on the real dataset it is as unavailable as the error column, and on the fixture it largely agrees with it. The error column carries the argument without it.
- **Colour constancy and white balance.** The pipeline treats RGB channels independently, exactly as the source does. That is wrong for a colour cast and out of scope for a critique of the luminance claim.
- **Confidence intervals on section 5.** The crossover is a single-seed measurement on one synthetic pattern. The direction and magnitude of the effect are large and consistent across the sweep; the precise crossover point is not resolved at this sample size, which is why the claim is stated as "above noise 4" rather than as a threshold.
- **The other coursework topics.** `legacy/` also contains anomaly detection, changepoint detection and Shewhart control charts from the original scripts. They belong with the unsupervised-learning repository, not here.

## License

MIT — see [LICENSE](LICENSE).
