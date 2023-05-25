import os

import fire
import pandas as pd

from luna.common.models import StoredTileSchema
from luna.pathology.cli.save_tiles import cli


def test_save_cli(tmp_path, dask_client):
    fire.Fire(
        cli,
        [
            "--slide_urlpath",
            "tests/testdata/pathology/123.svs",
            "--tile-size",
            "256",
            "--requested-magnification",
            "10",
            "--output_urlpath",
            str(tmp_path),
            "--batch_size",
            "16",
        ],
    )

    assert os.path.exists(f"{tmp_path}/123.tiles.parquet")

    StoredTileSchema.validate(pd.read_parquet(f"{tmp_path}/123.tiles.parquet"))
