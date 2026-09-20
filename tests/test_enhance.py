"""Point operations, and the theorem they exist to demonstrate."""

from itertools import pairwise

import pytest

from lowlight.enhance import (
    IDENTITY,
    Gain,
    apply_table,
    brightness,
    brightness_table,
    compose,
    contrast_table,
    gamma,
    gamma_table,
    stretch_table,
)
from lowlight.raster import MAX_VALUE, RasterError, grey
from lowlight.stats import distinct_levels, entropy_bits
from lowlight.synthetic import dark_pair


def test_every_table_has_256_entries():
    for table in (
        gamma_table(0.4),
        brightness_table(1.5),
        contrast_table(1.2),
        stretch_table(3, 40),
        IDENTITY,
    ):
        assert len(table) == 256


def test_gamma_one_is_the_identity():
    assert gamma_table(1.0) == IDENTITY


def test_gamma_below_one_lifts_the_shadows():
    table = gamma_table(0.4)
    assert table[10] > 10
    assert table[0] == 0
    assert table[255] == 255


def test_gamma_above_one_deepens_the_shadows():
    assert gamma_table(2.0)[128] < 128


def test_gamma_is_monotone_non_decreasing():
    table = gamma_table(0.3)
    assert all(a <= b for a, b in pairwise(table))


def test_non_positive_gamma_is_refused():
    with pytest.raises(RasterError, match="gamma must be positive"):
        gamma_table(0.0)


def test_brightness_multiplies_and_saturates():
    table = brightness_table(2.0)
    assert table[10] == 20
    assert table[200] == MAX_VALUE


def test_negative_brightness_is_refused():
    with pytest.raises(RasterError, match="non-negative"):
        brightness_table(-1.0)


def test_contrast_leaves_the_pivot_untouched():
    assert contrast_table(1.8, pivot=100.0)[100] == 100


def test_contrast_pushes_away_from_the_pivot():
    table = contrast_table(2.0, pivot=128.0)
    assert table[150] > 150
    assert table[100] < 100


def test_negative_contrast_is_refused():
    with pytest.raises(RasterError, match="non-negative"):
        contrast_table(-0.5)


def test_stretch_maps_the_bounds_to_the_ends():
    table = stretch_table(10, 20, headroom=1.0)
    assert table[10] == 0
    assert table[20] == MAX_VALUE


def test_stretch_clips_outside_the_bounds():
    table = stretch_table(10, 20)
    assert table[5] == 0
    assert table[250] == table[20]


def test_stretch_headroom_leaves_the_top_of_the_range_unused():
    assert stretch_table(0, 255, headroom=0.95)[255] == pytest.approx(242, abs=1)


def test_a_flat_image_stretches_to_the_identity_instead_of_dividing_by_zero():
    assert stretch_table(40, 40) == IDENTITY
    assert stretch_table(40, 10) == IDENTITY


def test_stretch_bounds_outside_the_byte_range_are_refused():
    with pytest.raises(RasterError, match="bounds"):
        stretch_table(-1, 20)


def test_compose_of_nothing_is_the_identity():
    assert compose() == IDENTITY


def test_compose_matches_applying_the_tables_in_turn():
    image = grey(4, 1, [3, 40, 120, 250])
    first, second = gamma_table(0.4), brightness_table(1.3)
    chained = apply_table(apply_table(image, first), second)
    composed = apply_table(image, compose(first, second))
    assert chained.data == composed.data


def test_compose_is_order_sensitive():
    first, second = gamma_table(0.3), stretch_table(5, 60)
    assert compose(first, second) != compose(second, first)


def test_compose_refuses_a_malformed_table():
    with pytest.raises(RasterError, match="256 entries"):
        compose((0, 1, 2))


def test_no_point_operation_can_increase_the_level_count():
    # The theorem. A lookup table is a function on 256 levels, so it merges
    # levels or keeps them, never splits them.
    dark = dark_pair(size=32).dark
    before = distinct_levels(dark.luma())
    for table in (
        gamma_table(0.3),
        brightness_table(2.0),
        contrast_table(1.8),
        stretch_table(1, 17),
    ):
        assert distinct_levels(apply_table(dark, table).luma()) <= before


def test_no_chain_of_point_operations_can_increase_the_level_count():
    dark = dark_pair(size=32).dark
    before = distinct_levels(dark.luma())
    chain = compose(
        gamma_table(0.3), brightness_table(1.8), contrast_table(1.5), stretch_table(1, 40)
    )
    assert distinct_levels(apply_table(dark, chain).luma()) <= before


def test_no_point_operation_can_increase_entropy():
    dark = dark_pair(size=32).dark
    before = entropy_bits(dark.luma())
    brightened = apply_table(dark, brightness_table(4.0))
    assert entropy_bits(brightened.luma()) <= before + 1e-12


def test_gamma_on_a_dark_image_preserves_entropy_exactly_when_it_merges_nothing():
    # Lifting a dark image is injective here, so not one bit is gained and
    # not one is lost: the "+664% luminance" is a relabelling.
    dark = dark_pair(size=32).dark
    assert entropy_bits(gamma(dark, 0.3).luma()) == pytest.approx(
        entropy_bits(dark.luma())
    )


def test_a_constant_gain_is_one_multiply_and_adds_no_levels():
    dark = dark_pair(size=32).dark
    gained = Gain(20.0).apply(dark)
    assert distinct_levels(gained.luma()) <= distinct_levels(dark.luma())


def test_the_convenience_wrappers_match_their_tables():
    image = grey(2, 1, [10, 100])
    assert gamma(image, 0.5).data == apply_table(image, gamma_table(0.5)).data
    assert brightness(image, 1.5).data == apply_table(image, brightness_table(1.5)).data
