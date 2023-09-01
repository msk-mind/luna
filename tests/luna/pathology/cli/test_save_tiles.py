import os

import fire
import pandas as pd

from luna.common.models import StoredTileSchema
from luna.pathology.cli.save_tiles import cli


def test_save_cli_s3(tmp_path, dask_client, s3fs_client):
    s3fs_client.mkdirs("mybucket", exist_ok=True)
    s3fs_client.put("tests/testdata/pathology/123.svs", "mybucket/test/")
    s3fs_client.put(
        "tests/testdata/pathology/generate_tiles/123/123.tiles.parquet",
        "mybucket/test/",
    )
    fire.Fire(
        cli,
        [
            "--slide_urlpath",
            "s3://mybucket/test/123.svs",
            "--tiles_urlpath",
            "s3://mybucket/test/123.tiles.parquet",
            "--output_urlpath",
            "s3://mybucket/test",
            "--storage_options",
            "{'key': '', 'secret': '', 'client_kwargs': {'endpoint_url': '"
            + s3fs_client.client_kwargs["endpoint_url"]
            + "'}}",
            "--batch_size",
            "16",
        ],
    )
    assert s3fs_client.exists("mybucket/test/123.tiles.parquet")


def test_save_cli(tmp_path, dask_client):
    fire.Fire(
        cli,
        [
            "--slide_urlpath",
            "tests/testdata/pathology/123.svs",
            "--tiles_urlpath",
            "tests/testdata/pathology//generate_tiles/123/123.tiles.parquet",
            "--output_urlpath",
            str(tmp_path),
            "--batch_size",
            "16",
        ],
    )

    assert os.path.exists(f"{tmp_path}/123.tiles.parquet")

    StoredTileSchema.validate(pd.read_parquet(f"{tmp_path}/123.tiles.parquet"))
