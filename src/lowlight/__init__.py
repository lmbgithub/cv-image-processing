"""Low-light image enhancement, and an honest account of what it recovers.

The module boundaries, and what each is allowed to know:

* `raster`    — 8-bit samples and geometry. Knows nothing about enhancement.
* `pnm`       — Netpbm I/O, so a real photograph needs no dependency.
* `stats`     — measurements only. Never modifies an image.
* `enhance`   — point operations, each expressed as a 256-entry lookup table.
* `filters`   — neighbourhood operations: Gaussian, median, unsharp mask.
* `synthetic` — a dark capture whose bright original is known.
* `pipeline`  — the source report's five adaptive stages, reproduced.
* `report`    — the comparisons the source report is missing.

`stats` importing nothing from `enhance` is the load-bearing boundary: the
measurements cannot be tuned to flatter the pipeline because they do not know
it exists.
"""

from lowlight.raster import MAX_VALUE, Raster, RasterError, grey, rgb

__all__ = ["MAX_VALUE", "Raster", "RasterError", "grey", "rgb"]
