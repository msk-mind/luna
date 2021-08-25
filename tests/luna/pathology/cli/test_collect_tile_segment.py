from click.testing import CliRunner

from luna.pathology.cli.collect_tile_segment import cli
import shutil, os

def test_cli():

    runner = CliRunner()
    result = runner.invoke(cli, [
        '-a', 'tests/luna/pathology/cli/testdata/test_config.yml',
        '-s', '123',
        '-m', 'tests/luna/pathology/cli/testdata/collect_tile_results.yml'])

    assert result.exit_code == 0
    assert os.path.exists('tests/luna/pathology/cli/testdata/data/test/slides/ovarian_clf_v1/123.parquet')

    # cleanup
    shutil.rmtree('tests/luna/pathology/cli/testdata/data/test/slides/ovarian_clf_v1/')

