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


def test_cli_s3(s3fs_client):
    s3fs_client.mkdirs("infertile", exist_ok=True)
    s3fs_client.put(
        "tests/testdata/pathology/save_tiles/123/123.tiles.parquet", "infertile/test/"
    )
    fire.Fire(
        cli,
        [
            "--tiles-urlpath",
            "s3://infertile/test/123.tiles.parquet",
            "--insecure",
            "--output-urlpath",
            "s3://infertile/out",
            "--torch-model-repo-or-dir",
            "tests/testdata/pathology/testhub",
            "--model-name",
            "test_custom_model",
            "--storage_options",
            "{'key': '', 'secret': '', 'client_kwargs': {'endpoint_url': '"
            + s3fs_client.client_kwargs["endpoint_url"]
            + "'}}",
        ],
    )

    assert s3fs_client.exists("s3://infertile/out/123.tiles.parquet")


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
