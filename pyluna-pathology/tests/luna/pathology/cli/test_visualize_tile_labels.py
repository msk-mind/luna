import shutil
from click.testing import CliRunner

from luna.pathology.cli.visualize_tile_labels import cli

def test_cli():

    runner = CliRunner()
    result = runner.invoke(cli, [
        '-a', 'pyluna-pathology/tests/luna/pathology/cli/testdata/test_config.yml',
        '-s', '123',
        '-m', 'pyluna-pathology/tests/luna/pathology/cli/testdata/visualize_tile_labels.yml'])

    assert result.exit_code == 0
    shutil.rmtree('pyluna-pathology/tests/luna/pathology/cli/testdata/data/test/slides/123/test_visualize_tiles')
