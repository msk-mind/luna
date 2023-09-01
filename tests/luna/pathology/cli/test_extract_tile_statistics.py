import os

import fire
import pandas as pd

from luna.pathology.cli.extract_tile_statistics import cli


def test_cli_extract_tile_statistics(tmp_path):
    fire.Fire(
        cli,
        [
            "--tiles-urlpath",
            "tests/testdata/pathology/test_tile_stats.parquet",
            "--output-urlpath",
            str(tmp_path),
        ],
    )

    assert os.path.exists(f"{tmp_path}/test_tile_stats_tile_stats.parquet")
    df = pd.read_parquet(f"{tmp_path}/test_tile_stats_tile_stats.parquet")
    cols = df.columns
    for col in [
        "Centroid X µm_nobs",
        "Centroid X µm_mean",
        "Centroid X µm_variance",
        "Centroid X µm_skewness",
        "Centroid X µm_kurtosis",
        "Centroid X µm_pct0",
        "Centroid X µm_pct25",
        "Centroid X µm_pct50",
        "Centroid X µm_pct75",
        "Centroid X µm_pct100",
    ]:
        assert col in cols


def test_cli_extract_tile_statistics_s3(s3fs_client):
    s3fs_client.mkdirs("tiletest", exist_ok=True)
    s3fs_client.put(
        "tests/testdata/pathology/test_tile_stats.parquet", "tiletest/test/"
    )
    fire.Fire(
        cli,
        [
            "--tiles-urlpath",
            "s3://tiletest/test/test_tile_stats.parquet",
            "--output-urlpath",
            "s3://tiletest/out/",
            "--storage_options",
            "{'key': '', 'secret': '', 'client_kwargs': {'endpoint_url': '"
            + s3fs_client.client_kwargs["endpoint_url"]
            + "'}}",
        ],
    )

    assert s3fs_client.exists("s3://tiletest/out/test_tile_stats_tile_stats.parquet")
