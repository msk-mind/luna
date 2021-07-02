import shutil
from click.testing import CliRunner

from data_processing.pathology.cli.generate_tile_labels import cli


def test_cli():

    runner = CliRunner()
    result = runner.invoke(cli, [
        '-a', 'tests/data_processing/pathology/cli/testdata/test_config.yaml',
        '-s', '123',
        '-m', 'tests/data_processing/pathology/cli/testdata/generate_tile_labels_with_ov_labels.json'])

    assert result.exit_code == 0
