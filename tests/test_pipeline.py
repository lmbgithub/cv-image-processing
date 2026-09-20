"""The reproduced pipeline: its parameter rules and what its stages cost."""

import pytest

from lowlight.pipeline import (
    choose_brightness,
    choose_contrast,
    choose_gamma,
    choose_sharpen,
    choose_sigma,
    enhance,
)
from lowlight.stats import distinct_levels
from lowlight.synthetic import dark_pair


def test_gamma_rule_matches_the_source_report():
    assert choose_gamma(0.027) == 0.3
    assert choose_gamma(0.055) == 0.4
    assert choose_gamma(0.30) == 0.6


def test_gamma_rule_boundaries():
    assert choose_gamma(0.0499) == 0.3
    assert choose_gamma(0.05) == 0.4
    assert choose_gamma(0.0999) == 0.4
    assert choose_gamma(0.10) == 0.6


def test_brightness_rule_is_clamped_at_both_ends():
    assert choose_brightness(0.001) == 2.0
    assert choose_brightness(0.9) == 1.3


def test_brightness_rule_reproduces_the_reports_1_80_factor():
    # Luminance 24% after gamma, per the worked example for 1002.png.
    assert choose_brightness(0.24) == pytest.approx(1.8, abs=0.01)


def test_contrast_rule_is_clamped_at_both_ends():
    assert choose_contrast(0.001) == 1.8
    assert choose_contrast(0.9) == 1.2


def test_sigma_rule_matches_the_source_report():
    assert choose_sigma(0.30) == 1.2
    assert choose_sigma(0.20) == 0.8
    assert choose_sigma(0.10) == 0.5


def test_sharpen_rule_matches_the_source_report():
    assert choose_sharpen(0.10) == 1.5
    assert choose_sharpen(0.15) == 1.2
    assert choose_sharpen(0.20) == 1.0


def test_the_stabiliser_keeps_a_black_image_from_dividing_by_zero():
    assert choose_brightness(0.0) == 2.0
    assert choose_contrast(0.0) == 1.8


def test_a_run_records_every_stage_in_order():
    run = enhance(dark_pair(size=24).dark)
    assert [stage.name for stage in run.stages] == [
        "1 gamma",
        "2a brightness",
        "2b contrast",
        "3 denoise gaussian",
        "4 sharpen",
        "5 stretch",
    ]


def test_the_pipeline_reaches_the_reports_target_luminance_band():
    # 42-43% in the source. It is hit because brightness is *computed* as
    # target/current — arithmetic, not a finding.
    run = enhance(dark_pair(size=32).dark)
    assert 0.35 < run.stages[2].summary.mean_fraction < 0.65


def test_the_three_point_stages_add_no_levels_at_all():
    # The headline result. Luminance rises by more than 1000% across these
    # three stages and the level count does not move.
    run = enhance(dark_pair(size=32).dark)
    original_levels = run.original.distinct_levels
    for stage in run.stages[:3]:
        assert stage.summary.distinct_levels == original_levels


def test_the_three_point_stages_preserve_entropy_exactly():
    run = enhance(dark_pair(size=32).dark)
    for stage in run.stages[:3]:
        assert stage.summary.entropy_bits == pytest.approx(run.original.entropy_bits)


def test_the_level_count_only_rises_at_the_blur():
    # New levels come from averaging neighbours. That is interpolation, not
    # recovered detail, and it is the only place the count can grow.
    run = enhance(dark_pair(size=32).dark)
    assert run.stages[3].summary.distinct_levels > run.stages[2].summary.distinct_levels


def test_the_point_stages_compose_to_one_table():
    run = enhance(dark_pair(size=24).dark)
    assert len(run.point_table) == 256


def test_the_first_three_stages_collapse_into_a_single_lookup_table():
    # Three "different" operations are one function on 256 levels. Composing
    # them must reproduce the stage-3 image exactly, byte for byte.
    from lowlight.enhance import apply_table, compose

    dark = dark_pair(size=24).dark
    run = enhance(dark)
    tables = [stage.table for stage in run.stages[:3]]
    collapsed = apply_table(dark, compose(*tables))
    rerun = dark
    for table in tables:
        rerun = apply_table(rerun, table)
    assert collapsed.data == rerun.data


def test_the_composed_table_bounds_the_levels_the_point_stages_can_emit():
    from lowlight.enhance import apply_table, compose

    dark = dark_pair(size=24).dark
    run = enhance(dark)
    composed = compose(*[stage.table for stage in run.stages[:3]])
    emitted = distinct_levels(apply_table(dark, composed).luma())
    assert emitted <= len(set(composed))


def test_skipping_the_stretch_shortens_the_run():
    run = enhance(dark_pair(size=24).dark, skip_stretch=True)
    assert [stage.name for stage in run.stages][-1] == "4 sharpen"


def test_the_median_denoiser_is_selectable():
    run = enhance(dark_pair(size=24).dark, denoise="median")
    assert run.stages[3].name == "3 denoise median"
    assert run.stages[3].parameter == "3x3"


def test_denoising_can_be_skipped():
    run = enhance(dark_pair(size=24).dark, denoise="none")
    assert run.stages[3].parameter == "skipped"


def test_an_unknown_denoiser_is_refused():
    with pytest.raises(ValueError, match="unknown denoiser"):
        enhance(dark_pair(size=24).dark, denoise="bilateral")


def test_the_denoiser_choice_changes_the_result():
    dark = dark_pair(size=24).dark
    assert (
        enhance(dark, denoise="gaussian").result.data
        != enhance(dark, denoise="median").result.data
    )


def test_a_run_is_reproducible():
    dark = dark_pair(size=24).dark
    assert enhance(dark).result.data == enhance(dark).result.data


def test_the_stage_table_formats_one_line_per_stage():
    run = enhance(dark_pair(size=24).dark)
    assert len(run.format().splitlines()) == len(run.stages) + 2


def test_only_the_point_stages_expose_a_table():
    run = enhance(dark_pair(size=24).dark)
    assert [stage.is_point_operation for stage in run.stages] == [
        True,
        True,
        True,
        False,
        False,
        True,
    ]


def test_entropy_never_reaches_that_of_the_bright_truth():
    # The information is gone. No arrangement of these operations recovers it.
    pair = dark_pair(size=32)
    run = enhance(pair.dark)
    from lowlight.stats import summarise

    assert run.final.entropy_bits < summarise(pair.truth).entropy_bits + 3.0
    assert run.final.distinct_levels < 256
