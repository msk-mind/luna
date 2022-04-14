import os
from click.testing import CliRunner

from luna.pathology.cli.run_tissue_detection import cli


def test_otsu(tmp_path):
    runner = CliRunner()
    result = runner.invoke(cli, [
        'pyluna-pathology/tests/luna/pathology/cli/testdata/data/123.svs',
        'pyluna-pathology/tests/luna/pathology/cli/testdata/data/generate_tiles/123/123.tiles.parquet',
        '-o', tmp_path,
        '-rmg', 5,
        '-nc', 1,
        '-fq', 'otsu_score > 0.5'])

    assert result.exit_code == 0
    assert os.path.exists(f"{tmp_path}/123-filtered.tiles.parquet")
    assert os.path.exists(f"{tmp_path}/metadata.yml")

def test_stain(tmp_path):
    runner = CliRunner()
    result = runner.invoke(cli, [
        'pyluna-pathology/tests/luna/pathology/cli/testdata/data/123.svs',
        'pyluna-pathology/tests/luna/pathology/cli/testdata/data/generate_tiles/123/123.tiles.parquet',
        '-o', tmp_path,
        '-rmg', 5,
        '-nc', 1,
        '-fq', 'stain0_score > 0.05'])

    assert result.exit_code == 0
    assert os.path.exists(f"{tmp_path}/123-filtered.tiles.parquet")
    assert os.path.exists(f"{tmp_path}/metadata.yml")
