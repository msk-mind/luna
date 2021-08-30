from click.testing import CliRunner
import os, shutil

from luna.pathology.cli.load_slide import cli


def test_cli():

    runner = CliRunner()
    result = runner.invoke(cli, [
        '-a', 'pyluna-pathology/tests/luna/pathology/cli/testdata/test_config.yml',
        '-s', '123',
        '-m', 'pyluna-pathology/tests/luna/pathology/cli/testdata/load_slide.yml'])

    assert result.exit_code == 0
    assert os.path.lexists('pyluna-pathology/tests/luna/pathology/cli/testdata/data/test/slides/123/pathology.etl/WholeSlideImage/data')
    assert os.path.exists('pyluna-pathology/tests/luna/pathology/cli/testdata/data/test/slides/123/pathology.etl/WholeSlideImage/metadata.json')

    # clean up
    shutil.rmtree("pyluna-pathology/tests/luna/pathology/cli/testdata/data/test/slides/123/pathology.etl/WholeSlideImage")


def test_cli_with_patientid():

    runner = CliRunner()
    result = runner.invoke(cli, [
        '-a', 'pyluna-pathology/tests/luna/pathology/cli/testdata/test_config.yml',
        '-s', '123',
        '-m', 'pyluna-pathology/tests/luna/pathology/cli/testdata/load_slide_with_patientid.yml'])

    assert result.exit_code == 0
    assert os.path.lexists('pyluna-pathology/tests/luna/pathology/cli/testdata/data/test/slides/123/pathology.etl/WholeSlideImage/data')
    assert os.path.exists('pyluna-pathology/tests/luna/pathology/cli/testdata/data/test/slides/123/pathology.etl/WholeSlideImage/metadata.json')
