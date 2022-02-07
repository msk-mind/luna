from click.testing import CliRunner

from luna.pathology.cli.infer_tile_labels import cli

import pandas as pd
from luna.pathology.schemas import SlideTiles

def test_cli(tmp_path):

    runner = CliRunner()

    result = runner.invoke(cli, [
            'pyluna-pathology/tests/luna/pathology/cli/testdata/data/generate_tiles/123/',
            '-o', tmp_path,
            '-rn', 'pyluna-pathology/tests/luna/pathology/cli/testdata/data/testhub',
            '-mn', 'test_custom_model',
            ])

    assert result.exit_code == 0
    assert SlideTiles.check(f"{tmp_path}/tile_scores_and_labels_pytorch_inference.csv")

    assert pd.read_csv(f"{tmp_path}/tile_scores_and_labels_pytorch_inference.csv").shape == (12, 14)

def test_cli_kwargs(tmp_path):

    runner = CliRunner()

    result = runner.invoke(cli, [
            'pyluna-pathology/tests/luna/pathology/cli/testdata/data/generate_tiles/123/',
            '-o', tmp_path,
            '-rn', 'pyluna-pathology/tests/luna/pathology/cli/testdata/data/testhub',
            '-mn', 'test_custom_model',
            '-kw', '{"n_channels":10}'
            ])

    assert result.exit_code == 0
    assert SlideTiles.check(f"{tmp_path}/tile_scores_and_labels_pytorch_inference.csv")

    assert pd.read_csv(f"{tmp_path}/tile_scores_and_labels_pytorch_inference.csv").shape == (12, 16)

def test_cli_resnet(tmp_path):

    runner = CliRunner()

    result = runner.invoke(cli, [
            'pyluna-pathology/tests/luna/pathology/cli/testdata/data/generate_tiles/123/',
            '-o', tmp_path,
            '-rn', 'pyluna-pathology/tests/luna/pathology/cli/testdata/data/testhub',
            '-mn', 'test_resnet',
            '-kw', '{"depth": 18, "pretrained":true}'
            ])

    assert result.exit_code == 0
    assert SlideTiles.check(f"{tmp_path}/tile_scores_and_labels_pytorch_inference.csv")

    assert pd.read_csv(f"{tmp_path}/tile_scores_and_labels_pytorch_inference.csv").shape == (12, 1006)



