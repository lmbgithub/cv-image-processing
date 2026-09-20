"""The comparisons, and what they establish."""

import pytest

from lowlight import report
from lowlight.pipeline import enhance
from lowlight.synthetic import dark_pair


def test_against_gain_returns_three_rows():
    rows = report.against_gain(dark_pair(size=24).dark)
    assert next(row.label for row in rows) == "dark original"
    assert len(rows) == 3


def test_the_dark_original_scores_zero_improvement_against_itself():
    rows = report.against_gain(dark_pair(size=24).dark)
    assert rows[0].luminance_gain_percent == pytest.approx(0.0, abs=1e-6)


def test_a_constant_multiply_scores_as_well_as_the_pipeline_on_the_headline_metric():
    # The point of the whole repository. One multiplication, no information
    # added, and "luminance improvement" cannot tell the two apart.
    _, gained, pipeline = report.against_gain(dark_pair(size=32).dark)
    assert gained.luminance_gain_percent > pipeline.luminance_gain_percent * 0.9


def test_the_constant_multiply_adds_no_levels_while_the_headline_soars():
    dark_summary, gained, _ = report.against_gain(dark_pair(size=32).dark)
    assert gained.luminance_gain_percent > 1000
    assert gained.summary.distinct_levels <= dark_summary.summary.distinct_levels


def test_the_constant_multiply_clips_where_the_pipeline_does_not():
    # The cost the headline metric does not charge for: 12% of the image
    # pinned to white and unrecoverable.
    _, gained, pipeline = report.against_gain(dark_pair(size=32).dark)
    assert gained.summary.clipped_high > 0.05
    assert pipeline.summary.clipped_high < gained.summary.clipped_high


def test_an_explicit_gain_factor_is_honoured():
    rows = report.against_gain(dark_pair(size=24).dark, factor=4.0)
    assert "x4.0" in rows[1].label


def test_against_truth_includes_the_bright_original():
    rows = report.against_truth(*_pair())
    assert [row.label for row in rows] == [
        "bright truth",
        "dark capture",
        "pipeline output",
    ]


def test_the_pipeline_reports_more_entropy_than_the_truth_it_is_approximating():
    # Not a typo and not a win. The blur and sharpen stages manufacture
    # interpolated levels, so entropy rises above the ground truth's own
    # while the image moves further away from it. Entropy resists point
    # operations and is inflated by neighbourhood ones, so it is a diagnostic,
    # never a quality score.
    truth, dark = _pair()
    rows = report.against_truth(truth, dark)
    assert rows[2].summary.entropy_bits > rows[0].summary.entropy_bits
    fidelities = report.fidelity(truth, dark)
    assert fidelities[2].error > fidelities[0].error


def test_at_low_noise_the_pipeline_is_further_from_the_truth_than_doing_nothing():
    pair = dark_pair(size=32, noise=0.0)
    rows = report.fidelity(pair.truth, pair.dark)
    assert rows[2].error > rows[0].error


def test_at_high_noise_the_pipeline_beats_doing_nothing():
    # The crossover. The pipeline earns its place in one regime, and the
    # metrics the source report used improve in both.
    pair = dark_pair(size=32, noise=8.0)
    rows = report.fidelity(pair.truth, pair.dark)
    assert rows[2].error < rows[0].error


def test_fidelity_names_every_method():
    pair = dark_pair(size=24)
    labels = [row.label for row in report.fidelity(pair.truth, pair.dark)]
    assert labels[0] == "dark capture"
    assert labels[2] == "five-stage pipeline"
    assert "constant gain" in labels[1]


def test_the_fidelity_header_and_rows_line_up():
    pair = dark_pair(size=24)
    rows = report.fidelity(pair.truth, pair.dark)
    assert len(report.fidelity_header()) == len(report.fidelity_row(rows[0]))


def test_order_matters_returns_both_orderings():
    rows = report.order_matters(dark_pair(size=24).dark)
    assert [row.label for row in rows] == ["gamma then stretch", "stretch then gamma"]


def test_the_two_orderings_of_the_same_two_operations_disagree():
    rows = report.order_matters(dark_pair(size=32).dark)
    assert rows[0].summary.deviation != rows[1].summary.deviation


def test_the_level_ceiling_is_far_below_full_scale():
    # The four point stages together can emit a couple of dozen levels. The
    # image ends up showing many more, all of them created by averaging.
    run = enhance(dark_pair(size=32).dark)
    assert report.level_ceiling(run) < 64
    assert report.full_scale_levels() == 256


def test_the_final_image_shows_more_levels_than_the_point_stages_can_emit():
    run = enhance(dark_pair(size=32).dark)
    assert run.final.distinct_levels > report.level_ceiling(run)


def test_the_header_and_rows_line_up():
    rows = report.against_gain(dark_pair(size=24).dark)
    assert len(report.header()) == len(report.row(rows[0]))


def _pair():
    pair = dark_pair(size=32)
    return pair.truth, pair.dark
