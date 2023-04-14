import os

from click.testing import CliRunner

from luna.pathology.cli.save_tiles import cli
from luna.pathology.common.schemas import SlideTiles


def test_save_cli(tmp_path):
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "tests/testdata/pathology/123.svs",
            "tests/testdata/pathology/generate_tiles/123/123.tiles.parquet",
            "-o",
            tmp_path,
            "-bx",
            16,
            "-nc",
            1,
        ],
    )

    assert result.exit_code == 0
    assert os.path.exists(f"{tmp_path}/123.tiles.parquet")

    assert SlideTiles.check(f"{tmp_path}/123.tiles.parquet")