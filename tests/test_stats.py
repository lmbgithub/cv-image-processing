"""The measurements, including the two that can be gamed."""

import math

import pytest

from lowlight.raster import grey
from lowlight.stats import (
    clipped_fractions,
    deviation,
    distinct_levels,
    entropy_bits,
    histogram,
    matched_error,
    mean,
    mean_absolute_error,
    percentile,
    relative_change,
    summarise,
)


def test_histogram_has_one_bin_per_level():
    assert len(histogram([0])) == 256


def test_histogram_counts_levels_not_ranges():
    counts = histogram([5, 5, 7])
    assert counts[5] == 2
    assert counts[7] == 1
    assert counts[6] == 0


def test_mean_and_deviation_of_known_samples():
    assert mean([0, 10]) == 5.0
    assert deviation([0, 10]) == 5.0


def test_deviation_of_a_constant_image_is_zero():
    assert deviation([7] * 9) == 0.0


def test_mean_of_no_samples_is_refused():
    with pytest.raises(ValueError, match="no samples"):
        mean([])


def test_deviation_of_no_samples_is_refused():
    with pytest.raises(ValueError, match="no samples"):
        deviation([])


def test_percentile_returns_a_level_present_in_the_data():
    samples = [0, 1, 2, 3, 100]
    assert percentile(samples, 0.5) in samples


def test_percentile_bounds_are_the_extremes():
    samples = [4, 8, 12]
    assert percentile(samples, 0.0) == 4
    assert percentile(samples, 1.0) == 12


def test_percentile_is_nearest_rank_not_interpolated():
    # An interpolated 50th percentile of [0, 10] is 5, which is not in the
    # image and cannot be a stretch bound.
    assert percentile([0, 10], 0.5) in (0, 10)


def test_out_of_range_percentile_is_refused():
    with pytest.raises(ValueError, match="fraction"):
        percentile([1, 2], 1.5)


def test_percentile_of_no_samples_is_refused():
    with pytest.raises(ValueError, match="no samples"):
        percentile([], 0.5)


def test_distinct_levels_counts_occupied_levels():
    assert distinct_levels([3, 3, 3]) == 1
    assert distinct_levels([0, 1, 2, 2]) == 3


def test_entropy_of_a_constant_image_is_zero():
    assert entropy_bits([9] * 16) == 0.0


def test_entropy_of_two_equally_likely_levels_is_one_bit():
    assert entropy_bits([0, 255]) == pytest.approx(1.0)


def test_entropy_of_four_equally_likely_levels_is_two_bits():
    assert entropy_bits([0, 1, 2, 3]) == pytest.approx(2.0)


def test_entropy_is_bounded_by_log2_of_the_level_count():
    samples = [0, 0, 0, 1]
    assert entropy_bits(samples) <= math.log2(distinct_levels(samples))


def test_entropy_of_no_samples_is_refused():
    with pytest.raises(ValueError, match="no samples"):
        entropy_bits([])


def test_clipped_fractions_count_only_the_rails():
    low, high = clipped_fractions([0, 0, 128, 255])
    assert low == 0.5
    assert high == 0.25


def test_nothing_is_clipped_in_a_mid_range_image():
    assert clipped_fractions([1, 128, 254]) == (0.0, 0.0)


def test_clipping_of_no_samples_is_refused():
    with pytest.raises(ValueError, match="no samples"):
        clipped_fractions([])


def test_relative_change_reproduces_the_reports_percentage():
    # 5.5% to 42.0% is the "+664%" in the source report.
    assert relative_change(0.055, 0.42) * 100 == pytest.approx(663.6, abs=0.5)


def test_relative_change_is_huge_only_because_the_denominator_is_tiny():
    # Same absolute gain, ten times the starting point, a tenth the headline.
    small = relative_change(0.01, 0.11)
    large = relative_change(0.10, 0.20)
    assert small > large * 5


def test_relative_change_survives_a_zero_baseline():
    assert relative_change(0.0, 1.0) > 0


def test_summary_reports_every_measurement():
    summary = summarise(grey(2, 2, [0, 10, 20, 255]))
    assert summary.minimum == 0
    assert summary.maximum == 255
    assert summary.distinct_levels == 4
    assert summary.dynamic_range == 255
    assert summary.clipped_low == 0.25
    assert summary.clipped_high == 0.25


def test_mean_fraction_is_the_reports_percentage_figure():
    assert summarise(grey(1, 1, [128])).mean_fraction == pytest.approx(0.502, abs=0.001)


def test_summary_formats_on_one_line():
    assert "\n" not in summarise(grey(2, 2, [1, 2, 3, 4])).format()


def test_mean_absolute_error_of_identical_samples_is_zero():
    assert mean_absolute_error([1, 2, 3], [1, 2, 3]) == 0.0


def test_mean_absolute_error_is_the_mean_of_the_differences():
    assert mean_absolute_error([0, 0], [1, 3]) == 2.0


def test_mean_absolute_error_is_symmetric():
    assert mean_absolute_error([0, 5], [5, 0]) == mean_absolute_error([5, 0], [0, 5])


def test_mismatched_lengths_are_refused():
    with pytest.raises(ValueError, match="against"):
        mean_absolute_error([1], [1, 2])


def test_comparing_no_samples_is_refused():
    with pytest.raises(ValueError, match="no samples"):
        mean_absolute_error([], [])


def test_matched_error_ignores_a_pure_brightness_difference():
    # A halved image is a perfect reconstruction up to exposure, and the
    # fidelity metric must not charge it for that.
    truth = [40, 80, 120, 160]
    halved = [20, 40, 60, 80]
    assert matched_error(halved, truth) == pytest.approx(0.0, abs=1.0)


def test_matched_error_still_penalises_a_real_difference():
    truth = [40, 80, 120, 160]
    scrambled = [160, 120, 80, 40]
    assert matched_error(scrambled, truth) > 30


def test_matched_error_of_a_black_image_falls_back_to_the_truth_mean():
    assert matched_error([0, 0], [10, 30]) == 20.0


def test_matched_error_against_no_truth_is_refused():
    with pytest.raises(ValueError, match="no samples"):
        matched_error([1], [])
