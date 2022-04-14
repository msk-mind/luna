import os
from click.testing import CliRunner

from luna.pathology.cli.save_tiles import cli

from luna.pathology.common.schemas import SlideTiles


def test_save_cli(tmp_path):
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "pyluna-pathology/tests/luna/pathology/cli/testdata/data/123.svs",
            "pyluna-pathology/tests/luna/pathology/cli/testdata/data/generate_tiles/123/123.tiles.parquet",
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
