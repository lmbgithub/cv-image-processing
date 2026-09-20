"""The 8-bit container: validation, luma, lookup tables."""

import pytest

from lowlight.raster import (
    MAX_VALUE,
    Raster,
    RasterError,
    constant,
    grey,
    quantise,
    rgb,
)


def test_shape_is_height_width_channels():
    assert grey(3, 2, [0] * 6).shape == (2, 3, 1)


def test_sample_count_must_match_the_geometry():
    with pytest.raises(RasterError, match="expected 6 samples"):
        Raster(3, 2, 1, (0, 0, 0))


def test_non_positive_dimensions_are_refused():
    with pytest.raises(RasterError, match="dimensions must be positive"):
        Raster(0, 2, 1, ())


def test_only_one_or_three_channels_are_allowed():
    with pytest.raises(RasterError, match="channels must be 1 or 3"):
        Raster(1, 1, 2, (0, 0))


def test_samples_outside_the_byte_range_are_refused():
    with pytest.raises(RasterError, match=r"outside 0\.\.255"):
        Raster(1, 1, 1, (256,))
    with pytest.raises(RasterError, match=r"outside 0\.\.255"):
        Raster(1, 1, 1, (-1,))


def test_float_samples_are_refused():
    with pytest.raises(RasterError, match="must be int"):
        Raster(1, 1, 1, (12.5,))


def test_a_boolean_sample_is_refused_although_bool_is_an_int():
    # True == 1 in Python, so an unguarded check would accept this and store a
    # boolean in an image. It is a real bug class, not a hypothetical one.
    with pytest.raises(RasterError, match="must be int"):
        Raster(1, 1, 1, (True,))


def test_sample_out_of_range_raises_instead_of_wrapping():
    image = grey(2, 2, [1, 2, 3, 4])
    with pytest.raises(RasterError, match="outside"):
        image.sample(2, 0)
    with pytest.raises(RasterError, match="outside"):
        image.sample(0, -1)


def test_clamped_replicates_the_edge():
    image = grey(2, 2, [1, 2, 3, 4])
    assert image.clamped(-5, 0) == 1
    assert image.clamped(9, 9) == 4


def test_channel_out_of_range_is_refused():
    with pytest.raises(RasterError, match="channel"):
        grey(1, 1, [0]).sample(0, 0, 1)


def test_plane_extracts_one_channel():
    image = rgb(2, 1, [10, 20, 30, 40, 50, 60])
    assert image.plane(0) == (10, 40)
    assert image.plane(1) == (20, 50)
    assert image.plane(2) == (30, 60)


def test_luma_of_a_grey_image_is_the_image_itself():
    # Not a weighted sum of three identical planes, which would round
    # differently and make grey and RGB paths disagree.
    image = grey(2, 1, [7, 200])
    assert image.luma() == (7, 200)


def test_luma_uses_rec_601_weights():
    assert rgb(1, 1, [255, 0, 0]).luma() == (76,)
    assert rgb(1, 1, [0, 255, 0]).luma() == (150,)
    assert rgb(1, 1, [0, 0, 255]).luma() == (29,)


def test_luma_weights_sum_to_white():
    assert rgb(1, 1, [255, 255, 255]).luma() == (255,)


def test_to_grey_preserves_geometry():
    assert rgb(4, 3, [128] * 36).to_grey().shape == (3, 4, 1)


def test_quantise_rounds_and_clips():
    assert quantise(-3.0) == 0
    assert quantise(300.0) == MAX_VALUE
    assert quantise(12.5) == 12 or quantise(12.5) == 13  # banker's rounding
    assert quantise(12.4) == 12


def test_map_samples_applies_the_table():
    table = tuple(MAX_VALUE - v for v in range(MAX_VALUE + 1))
    assert grey(2, 1, [0, 255]).map_samples(table).data == (255, 0)


def test_a_table_of_the_wrong_size_is_refused():
    with pytest.raises(RasterError, match="256 entries"):
        grey(1, 1, [0]).map_samples((0, 1, 2))


def test_constant_fills_every_sample():
    assert set(constant(4, 4, 33).data) == {33}


def test_render_maps_black_and_white_to_the_end_levels():
    assert grey(2, 1, [0, 255]).render(levels=" #") == " #"


def test_raster_is_frozen():
    with pytest.raises(AttributeError):
        grey(1, 1, [0]).width = 5
