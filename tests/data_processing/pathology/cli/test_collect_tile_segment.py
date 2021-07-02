from click.testing import CliRunner

from data_processing.pathology.cli.collect_tile_segment import cli
import shutil, os

def test_cli():

    runner = CliRunner()
    result = runner.invoke(cli, [
        '-a', 'tests/data_processing/pathology/cli/testdata/test_config.yaml',
        '-s', '123',
        '-m', 'tests/data_processing/pathology/cli/testdata/collect_tile_results.json'])

    assert result.exit_code == 0
    assert os.path.exists('tests/data_processing/pathology/cli/testdata/data/test/slides/ovarian_clf_v1/123/123.parquet')

    # cleanup
    shutil.rmtree('tests/data_processing/pathology/cli/testdata/data/test/slides/ovarian_clf_v1/')

