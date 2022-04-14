import os
from click.testing import CliRunner

from luna.pathology.cli.generate_tiles import cli

from luna.pathology.common.schemas import SlideTiles


def test_cli(tmp_path):
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "pyluna-pathology/tests/luna/pathology/cli/testdata/data/123.svs",
            "-o",
            tmp_path,
            "-rts",
            256,
            "-rmg",
            10,
        ],
    )

    assert result.exit_code == 0
    assert os.path.exists(f"{tmp_path}/123.tiles.parquet")

    assert SlideTiles.check(f"{tmp_path}/123.tiles.parquet")


def test_cli_bad_mag(tmp_path):
    runner = CliRunner()

    result = runner.invoke(
        cli,
        [
            "pyluna-pathology/tests/luna/pathology/cli/testdata/data/123.svs",
            "-o",
            tmp_path,
            "-rts",
            256,
            "-rmg",
            3,
        ],
    )

    assert result.exit_code == 1
