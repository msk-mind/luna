from click.testing import CliRunner
import os, shutil

from data_processing.pathology.cli.load_slide import cli

def test_cli():

    runner = CliRunner()
    result = runner.invoke(cli, [
        '-a', 'tests/data_processing/pathology/cli/testdata/test_config.yaml',
        '-s', '123',
        '-m', 'tests/data_processing/pathology/cli/testdata/load_slide.json'])

    assert result.exit_code == 0
    assert os.path.lexists('tests/data_processing/pathology/cli/testdata/data/test/slides/123/pathology.etl/WholeSlideImage/data')
    assert os.path.exists('tests/data_processing/pathology/cli/testdata/data/test/slides/123/pathology.etl/WholeSlideImage/metadata.json')

    # clean up
    shutil.rmtree("tests/data_processing/pathology/cli/testdata/data/test/slides/123/pathology.etl/WholeSlideImage")

def test_cli_with_patientid():

    runner = CliRunner()
    result = runner.invoke(cli, [
        '-a', 'tests/data_processing/pathology/cli/testdata/test_config.yaml',
        '-s', '123',
        '-m', 'tests/data_processing/pathology/cli/testdata/load_slide_with_patientid.json'])

    assert result.exit_code == 0
    assert os.path.lexists('tests/data_processing/pathology/cli/testdata/data/test/slides/123/pathology.etl/WholeSlideImage/data')
    assert os.path.exists('tests/data_processing/pathology/cli/testdata/data/test/slides/123/pathology.etl/WholeSlideImage/metadata.json')
