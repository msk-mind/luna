import fire
import pandas as pd

from luna.common.models import TileSchema
from luna.pathology.cli.infer_tile_labels import cli


def test_cli(tmp_path):
    fire.Fire(
        cli,
        [
            "--tiles-urlpath",
            "tests/testdata/pathology/save_tiles/123/123.tiles.parquet",
            "--insecure",
            "--output-urlpath",
            str(tmp_path),
            "--torch-model-repo-or-dir",
            "tests/testdata/pathology/testhub",
            "--model-name",
            "test_custom_model",
        ],
    )
    # Default to 2 channels..
    df = pd.read_parquet(f"{tmp_path}/123.tiles.parquet")
    TileSchema.validate(df.reset_index())
    assert df.shape == (12, 9)

    assert set(["Background", "Tumor"]).intersection(set(df.columns)) == set(
        ["Background", "Tumor"]
    )


def test_cli_kwargs(tmp_path):
    fire.Fire(
        cli,
        [
            "--tiles-urlpath",
            "tests/testdata/pathology/save_tiles/123/123.tiles.parquet",
            "--insecure",
            "--output-urlpath",
            str(tmp_path),
            "--torch-model-repo-or-dir",
            "tests/testdata/pathology/testhub",
            "--model-name",
            "test_custom_model",
            "--kwargs",
            '{"n_channels":10}',
        ],
    )

    df = pd.read_parquet(f"{tmp_path}/123.tiles.parquet")
    TileSchema.validate(df.reset_index())

    assert df.shape == (12, 17)  # 8 more


def test_cli_resnet(tmp_path):
    fire.Fire(
        cli,
        [
            "--tiles-urlpath",
            "tests/testdata/pathology/save_tiles/123/123.tiles.parquet",
            "--output-urlpath",
            str(tmp_path),
            "--torch-model-repo-or-dir",
            "tests/testdata/pathology/testhub",
            "--model-name",
            "test_resnet",
            "--kwargs",
            '{"depth": 18, "pretrained": True}',
        ],
    )

    df = pd.read_parquet(f"{tmp_path}/123.tiles.parquet")

    TileSchema.validate(df.reset_index())
    assert df.shape == (12, 1007)
