import os
from click.testing import CliRunner

import pandas as pd
from luna.pathology.cli.extract_tile_statistics import cli


def test_cli_generate_mask(tmp_path):
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "pyluna-pathology/tests/luna/pathology/cli/testdata/test_tile_stats.csv",
            "-o",
            tmp_path,
        ],
    )

    assert result.exit_code == 0
    assert os.path.exists(f"{tmp_path}/test_tile_stats_tile_stats.csv")
    df = pd.read_csv(f"{tmp_path}/test_tile_stats_tile_stats.csv")
    cols = df.columns
    for col in [
        "Centroid X µm_nobs",
        "Centroid X µm_mean",
        "Centroid X µm_variance",
        "Centroid X µm_skewness",
        "Centroid X µm_kurtosis",
        "Centroid X µm_lognorm_fit_p0",
        "Centroid X µm_lognorm_fit_p2",
        "Centroid X µm_pct0",
        "Centroid X µm_pct25",
        "Centroid X µm_pct50",
        "Centroid X µm_pct75",
        "Centroid X µm_pct100",
    ]:
        assert col in cols