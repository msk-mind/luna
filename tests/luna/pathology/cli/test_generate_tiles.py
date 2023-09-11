import os

import fire
import pandas as pd
import pytest

from luna.common.models import TileSchema
from luna.pathology.cli.generate_tiles import cli


def test_cli_s3(s3fs_client, dask_client):
    s3fs_client.mkdirs("test2", exist_ok=True)
    fire.Fire(
        cli,
        [
            "--slide-urlpath",
            "tests/testdata/pathology/123.svs",
            "--output-urlpath",
            "s3://test2/test",
            "--tile-size",
            "256",
            "--requested-magnification",
            "10",
            "--output_storage_options",
            "{'key': '', 'secret': '', 'client_kwargs': {'endpoint_url': '"
            + s3fs_client.client_kwargs["endpoint_url"]
            + "'}}",
        ],
    )
    assert s3fs_client.exists("test2/test/123.tiles.parquet")


def test_cli(tmp_path, dask_client):
    fire.Fire(
        cli,
        [
            "--slide-urlpath",
            "tests/testdata/pathology/123.svs",
            "--output-urlpath",
            str(tmp_path),
            "--tile-size",
            "256",
            "--requested-magnification",
            "10",
        ],
    )

    assert os.path.exists(f"{tmp_path}/123.tiles.parquet")
    TileSchema.validate(pd.read_parquet(f"{tmp_path}/123.tiles.parquet"))


def test_cli_bad_mag(tmp_path):
    with pytest.raises(Exception):
        fire.Fire(
            cli,
            [
                "--slide-urlpath",
                "tests/testdata/pathology/123.svs",
                "--output-urlpath",
                str(tmp_path),
                "--tile-size",
                "256" "--requested-magnification",
                "3",
            ],
        )
