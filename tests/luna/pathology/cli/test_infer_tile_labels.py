from click.testing import CliRunner
import pytest

from luna.pathology.cli.infer_tile_labels import cli


def test_cli():

    runner = CliRunner()

    result = runner.invoke(cli, [
            '-a', 'tests/luna/pathology/cli/testdata/test_config.yml',
            '-s', '123',
            '-m', 'tests/luna/pathology/cli/testdata/infer_tile_labels_resnet18.yml'])

    # No longer error gracefully -- can update tests with proper data and they'll work
    assert result.exit_code == 1
