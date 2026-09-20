"""The fixture with known ground truth."""

import pytest

from lowlight.raster import MAX_VALUE, RasterError
from lowlight.stats import distinct_levels, summarise
from lowlight.synthetic import crush, dark_pair, noise_field, pattern


def test_pattern_spans_the_full_range():
    image = pattern(32)
    assert min(image.data) == 0
    assert max(image.data) == MAX_VALUE


def test_pattern_contains_a_one_pixel_line():
    image = pattern(32)
    row = image.height // 2
    assert all(image.sample(row, x) == MAX_VALUE for x in range(image.width))
    assert image.sample(row - 1, 0) != MAX_VALUE


def test_pattern_is_square_and_deterministic():
    assert pattern(24).shape == (24, 24, 1)
    assert pattern(24).data == pattern(24).data


def test_a_pattern_too_small_is_refused():
    with pytest.raises(RasterError, match="at least 16"):
        pattern(8)


def test_crushing_lowers_the_mean_by_the_exposure():
    bright = pattern(24)
    dark = crush(bright, 0.05)
    assert summarise(dark).mean < summarise(bright).mean * 0.06


def test_crushing_destroys_levels_irreversibly():
    # The step that makes low-light photography hard: 8-bit requantisation.
    bright = pattern(24)
    assert distinct_levels(crush(bright, 0.05).luma()) < distinct_levels(bright.luma())


def test_crushing_without_noise_is_deterministic():
    bright = pattern(24)
    assert crush(bright, 0.05).data == crush(bright, 0.05).data


def test_crushing_with_noise_is_reproducible_from_the_seed():
    bright = pattern(24)
    assert (
        crush(bright, 0.05, seed=3, noise=2.0).data
        == crush(bright, 0.05, seed=3, noise=2.0).data
    )


def test_different_seeds_give_different_noise():
    bright = pattern(24)
    assert (
        crush(bright, 0.05, seed=3, noise=2.0).data
        != crush(bright, 0.05, seed=4, noise=2.0).data
    )


def test_full_exposure_without_noise_is_the_identity():
    bright = pattern(24)
    assert crush(bright, 1.0).data == bright.data


def test_an_exposure_outside_the_unit_interval_is_refused():
    with pytest.raises(RasterError, match="exposure"):
        crush(pattern(16), 0.0)
    with pytest.raises(RasterError, match="exposure"):
        crush(pattern(16), 1.5)


def test_negative_noise_is_refused():
    with pytest.raises(RasterError, match="noise"):
        crush(pattern(16), 0.5, noise=-1.0)


def test_dark_pair_keeps_both_halves_of_the_ground_truth():
    pair = dark_pair(size=24)
    assert pair.truth.shape == pair.dark.shape
    assert summarise(pair.dark).mean_fraction < 0.1
    assert summarise(pair.truth).mean_fraction > 0.3


def test_dark_pair_records_its_exposure_and_seed():
    pair = dark_pair(size=24, exposure=0.04, seed=5)
    assert pair.exposure == 0.04
    assert pair.seed == 5


def test_noise_field_has_no_structure_worth_measuring():
    field = noise_field(24, level=8)
    assert summarise(field).mean_fraction < 0.1


def test_noise_field_is_reproducible():
    assert noise_field(16, seed=2).data == noise_field(16, seed=2).data
