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
