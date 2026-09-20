"""The command line: works with no dataset, writes what it claims."""

import pytest

from lowlight import pnm
from lowlight.cli import build_parser, main
from lowlight.synthetic import dark_pair


def test_it_runs_with_no_input_file(capsys):
    assert main(["--size", "24"]) == 0
    assert "synthetic fixture" in capsys.readouterr().out


def test_the_default_denoiser_is_gaussian():
    assert build_parser().parse_args([]).denoise == "gaussian"


def test_an_unknown_denoiser_is_rejected():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--denoise", "bilateral"])


def test_stages_flag_prints_every_stage(capsys):
    main(["--size", "24", "--stages"])
    out = capsys.readouterr().out
    for stage in ("1 gamma", "2a brightness", "3 denoise", "4 sharpen", "5 stretch"):
        assert stage in out


def test_compare_flag_prints_the_gain_baseline(capsys):
    main(["--size", "24", "--compare"])
    out = capsys.readouterr().out
    assert "constant gain" in out
    assert "levels the point stages can emit" in out


def test_order_flag_prints_both_orderings(capsys):
    main(["--size", "24", "--order"])
    out = capsys.readouterr().out
    assert "gamma then stretch" in out
    assert "stretch then gamma" in out


def test_it_reads_a_real_file(tmp_path, capsys):
    path = tmp_path / "dark.pgm"
    pnm.write(path, dark_pair(size=24).dark)
    assert main(["--input", str(path)]) == 0
    assert str(path) in capsys.readouterr().out


def test_it_writes_a_readable_file(tmp_path):
    out = tmp_path / "bright.pgm"
    main(["--size", "24", "--output", str(out)])
    assert pnm.read(out).shape == (24, 24, 1)


def test_the_written_file_is_the_enhanced_image_not_the_input(tmp_path):
    out = tmp_path / "bright.pgm"
    main(["--size", "24", "--output", str(out)])
    assert pnm.read(out).data != dark_pair(size=24).dark.data


def test_an_rgb_file_survives_the_round_trip(tmp_path):
    from lowlight.raster import rgb

    path = tmp_path / "colour.ppm"
    out = tmp_path / "out.ppm"
    pnm.write(path, rgb(16, 16, [5, 7, 9] * 256))
    main(["--input", str(path), "--output", str(out)])
    assert pnm.read(out).channels == 3
