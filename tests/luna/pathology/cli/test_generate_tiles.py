import os

import fire
import pandas as pd
import pytest

from luna.common.models import TileSchema
from luna.pathology.cli.generate_tiles import cli


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
