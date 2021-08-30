import os
from click.testing import CliRunner

from luna.pathology.cli.generate_tile_labels import cli


def test_cli():

    runner = CliRunner()
    result = runner.invoke(cli, [
        '-a', 'pyluna-pathology/tests/luna/pathology/cli/testdata/test_config.yml',
        '-s', '123',
        '-m', 'pyluna-pathology/tests/luna/pathology/cli/testdata/generate_tile_labels_with_ov_labels.yml'])

    assert result.exit_code == 0
    assert os.path.exists("pyluna-pathology/tests/luna/pathology/cli/testdata/data/test/slides/123/test_generate_tile_ov_labels/TileImages/data/address.slice.csv")
    assert os.path.exists("pyluna-pathology/tests/luna/pathology/cli/testdata/data/test/slides/123/test_generate_tile_ov_labels/TileImages/data/tiles.slice.pil")
    assert os.path.exists("pyluna-pathology/tests/luna/pathology/cli/testdata/data/test/slides/123/test_generate_tile_ov_labels/TileImages/data/metadata.json")
