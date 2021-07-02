from click.testing import CliRunner
import os

from data_processing.pathology.cli.load_slide import cli


def test_cli(mocker):

    runner = CliRunner()
    result = runner.invoke(cli, [
        '-a', 'tests/data_processing/pathology/cli/testdata/test_config.yaml',
        '-s', '123',
        '-m', 'tests/data_processing/pathology/cli/testdata/load_slide.json'])

    assert result.exit_code == 0
    assert os.path.lexists('tests/data_processing/pathology/cli/testdata/data/test/slides/123/pathology.etl/WholeSlideImage/data')
    assert os.path.exists('tests/data_processing/pathology/cli/testdata/data/test/slides/123/pathology.etl/WholeSlideImage/metadata.json')
