import os
from click.testing import CliRunner

import pandas as pd
from luna.pathology.cli.extract_kfunction_statistics import cli


def test_cli(tmp_path):
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "pyluna-pathology/tests/luna/pathology/cli/testdata/test_tile_stats.parquet",
            "-o",
            tmp_path,
            "-il",
            "Centroid X µm",
            "-r",
            160.0,
            "-rts",
            300,
            "-rtd",
            300,
        ],
    )

    assert result.exit_code == 0
    assert os.path.exists(f"{tmp_path}/test_tile_stats_kfunction_supertiles.parquet")

    df = pd.read_parquet(f"{tmp_path}/test_tile_stats_kfunction_supertiles.parquet")
    assert "ikfunction_r160.0_stainCentroid_X_µm" in df.columns
    assert df["ikfunction_r160.0_stainCentroid_X_µm_norm"].values[0] == 1.0
