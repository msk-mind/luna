from click.testing import CliRunner

from luna.pathology.cli.collect_tile_segment import cli
import shutil, os

def test_cli():

    runner = CliRunner()
    result = runner.invoke(cli, [
        '-a', 'pyluna-pathology/tests/luna/pathology/cli/testdata/test_config.yml',
        '-s', '123',
        '-m', 'pyluna-pathology/tests/luna/pathology/cli/testdata/collect_tile_results.yml'])

    assert result.exit_code == 0
    assert os.path.exists('pyluna-pathology/tests/luna/pathology/cli/testdata/data/test/slides/ovarian_clf_v1/123.parquet')

    # cleanup
    shutil.rmtree('pyluna-pathology/tests/luna/pathology/cli/testdata/data/test/slides/ovarian_clf_v1/')

