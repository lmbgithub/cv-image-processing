"""Gaussian against median, and the two facts that decide between them."""

import pytest

from lowlight.filters import (
    gaussian_blur,
    gaussian_kernel,
    median_filter,
    noise_estimate,
    sharpen,
)
from lowlight.raster import MAX_VALUE, RasterError, constant, grey
from lowlight.stats import deviation
from lowlight.synthetic import noise_field, pattern


def test_gaussian_kernel_sums_to_one():
    assert sum(gaussian_kernel(1.2)) == pytest.approx(1.0)


def test_gaussian_kernel_is_symmetric_and_peaks_at_the_centre():
    kernel = gaussian_kernel(1.0)
    assert kernel == list(reversed(kernel))
    assert kernel[len(kernel) // 2] == max(kernel)


def test_gaussian_kernel_radius_is_three_sigma():
    assert len(gaussian_kernel(1.0)) == 7
    assert len(gaussian_kernel(2.0)) == 13


def test_non_positive_sigma_is_refused():
    with pytest.raises(RasterError, match="sigma must be positive"):
        gaussian_kernel(0.0)


def test_blurring_a_flat_image_changes_nothing():
    flat = constant(8, 8, 70)
    assert gaussian_blur(flat, 1.0).data == flat.data


def test_blurring_preserves_geometry():
    assert gaussian_blur(pattern(16), 0.8).shape == (16, 16, 1)


def test_blurring_preserves_the_mean_of_a_flat_image_at_the_border():
    # Replicated edges, so no darkening. Zero padding would fail this.
    flat = constant(8, 8, 200)
    assert gaussian_blur(flat, 1.2).sample(0, 0) == 200


def test_blurring_reduces_noise():
    field = noise_field(24)
    assert noise_estimate(gaussian_blur(field, 0.8)) < noise_estimate(field)


def test_median_of_a_flat_image_changes_nothing():
    flat = constant(8, 8, 70)
    assert median_filter(flat, 3).data == flat.data


def test_median_deletes_a_one_pixel_line_entirely():
    # The fact the usual "median preserves edges" summary omits. A feature
    # thinner than half the window is never the middle of the sorted window.
    image = pattern(24)
    row = image.height // 2
    # Sampled where the line crosses the dark end of the gradient, so the
    # surroundings are dim and the loss is unambiguous: 255 becomes 22.
    assert image.sample(row, 1) == MAX_VALUE
    assert median_filter(image, 3).sample(row, 1) < 50


def test_a_gaussian_blur_keeps_part_of_that_line():
    image = pattern(24)
    row = image.height // 2
    assert gaussian_blur(image, 0.8).sample(row, 1) > 100


def test_median_preserves_a_step_edge():
    edge = grey(6, 1, [0, 0, 0, 255, 255, 255])
    filtered = median_filter(edge, 3)
    assert filtered.data[:3] == (0, 0, 0)
    assert filtered.data[3:] == (255, 255, 255)


def test_a_gaussian_blur_softens_that_same_edge():
    edge = grey(6, 1, [0, 0, 0, 255, 255, 255])
    assert 0 < gaussian_blur(edge, 1.0).sample(0, 2) < 255


def test_median_removes_an_isolated_hot_pixel():
    image = grey(3, 3, [10] * 4 + [255] + [10] * 4)
    assert median_filter(image, 3).sample(1, 1) == 10


def test_an_even_median_window_is_refused():
    with pytest.raises(RasterError, match="positive and odd"):
        median_filter(constant(4, 4, 0), 2)


def test_sharpen_leaves_a_flat_image_flat():
    # The kernel is normalised, so sharpening cannot brighten a flat field.
    # Skipping that division is the standard bug.
    flat = constant(8, 8, 100)
    assert set(sharpen(flat, 1.2).data) == {100}


def test_sharpen_increases_the_deviation_of_a_pure_noise_field():
    # "Contrast improved" on an image that contains no information at all.
    field = noise_field(24)
    before = deviation(field.luma())
    after = deviation(sharpen(field, 1.2).luma())
    assert after > before * 2


def test_sharpen_restores_most_of_the_noise_the_blur_removed():
    # Stage 4 of the pipeline partially undoes stage 3.
    field = noise_field(24)
    blurred = gaussian_blur(field, 0.8)
    resharpened = sharpen(blurred, 1.2)
    assert noise_estimate(resharpened) > noise_estimate(blurred) * 2


def test_non_positive_sharpen_strength_is_refused():
    with pytest.raises(RasterError, match="strength must be positive"):
        sharpen(constant(4, 4, 0), 0.0)


def test_noise_estimate_of_a_flat_image_is_zero():
    assert noise_estimate(constant(8, 8, 120)) == 0.0


def test_filters_work_on_rgb_without_mixing_the_channels():
    from lowlight.raster import rgb

    image = rgb(4, 4, [255, 0, 0] * 16)
    blurred = gaussian_blur(image, 1.0)
    assert set(blurred.plane(1)) == {0}
    assert set(blurred.plane(2)) == {0}
