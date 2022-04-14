from click.testing import CliRunner

from luna.pathology.cli.infer_tile_labels import cli

import pandas as pd
from luna.pathology.common.schemas import SlideTiles


def test_cli(tmp_path):
    runner = CliRunner()

    result = runner.invoke(
        cli,
        [
            "pyluna-pathology/tests/luna/pathology/cli/testdata/data/save_tiles/123/",
            "-o",
            tmp_path,
            "-rn",
            "pyluna-pathology/tests/luna/pathology/cli/testdata/data/testhub",
            "-mn",
            "test_custom_model",
        ],
    )

    assert result.exit_code == 0
    assert SlideTiles.check(
        f"{tmp_path}" f"/tile_scores_and_labels_pytorch_inference.parquet"
    )

    # Default to 2 channels..
    df = pd.read_parquet(f"{tmp_path}/tile_scores_and_labels_pytorch_inference.parquet")
    assert df.shape == (12, 9)

    assert set(["Background", "Tumor"]).intersection(set(df.columns)) == set(
        ["Background", "Tumor"]
    )


def test_cli_kwargs(tmp_path):
    runner = CliRunner()

    result = runner.invoke(
        cli,
        [
            "pyluna-pathology/tests/luna/pathology/cli/testdata/data/save_tiles/123/",
            "-o",
            tmp_path,
            "-rn",
            "pyluna-pathology/tests/luna/pathology/cli/testdata/data/testhub",
            "-mn",
            "test_custom_model",
            "-kw",
            '{"n_channels":10}',
        ],
    )

    assert result.exit_code == 0
    assert SlideTiles.check(
        f"{tmp_path}" f"/tile_scores_and_labels_pytorch_inference.parquet"
    )

    df = pd.read_parquet(f"{tmp_path}/tile_scores_and_labels_pytorch_inference.parquet")

    assert df.shape == (12, 17)  # 8 more


def test_cli_resnet(tmp_path):
    runner = CliRunner()

    result = runner.invoke(
        cli,
        [
            "pyluna-pathology/tests/luna/pathology/cli/testdata/data/save_tiles/123/",
            "-o",
            tmp_path,
            "-rn",
            "pyluna-pathology/tests/luna/pathology/cli/testdata/data/testhub",
            "-mn",
            "test_resnet",
            "-kw",
            '{"depth": 18, "pretrained": True}',
        ],
    )

    assert result.exit_code == 0
    assert SlideTiles.check(f"{tmp_path}/tile_scores_and_labels_pytorch_inference.parquet")

    assert pd.read_parquet(
        f"{tmp_path}/tile_scores_and_labels_pytorch_inference.parquet"
    ).shape == (12, 1007)

    assert pd.read_parquet(
        f"{tmp_path}/tile_scores_and_labels_pytorch_inference.parquet"
    ).shape == (12, 1007)
