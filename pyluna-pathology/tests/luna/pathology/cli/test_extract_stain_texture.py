import os
from click.testing import CliRunner

import pandas as pd
from luna.pathology.cli.extract_stain_texture import cli


def test_cli(tmp_path):
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "pyluna-pathology/tests/luna/pathology/cli/testdata/data/123.svs",
            "pyluna-pathology/tests/luna/pathology/cli/testdata/data/generate_mask/mask_full_res.tif",
            "-o",
            tmp_path,
            "-tx",
            500,
            "-sc",
            0,
            "-sf",
            10,
            "-glcm",
            "ClusterTendency",
        ],
    )

    assert result.exit_code == 0
    assert os.path.exists(f"{tmp_path}/stainomics.csv")

    df = pd.read_csv(f"{tmp_path}/stainomics.csv")
    assert set(
        [
            "pixel_original_glcm_ClusterTendency_channel_0_nobs",
            "pixel_original_glcm_ClusterTendency_channel_0_min",
            "pixel_original_glcm_ClusterTendency_channel_0_max",
            "pixel_original_glcm_ClusterTendency_channel_0_mean",
            "pixel_original_glcm_ClusterTendency_channel_0_variance",
            "pixel_original_glcm_ClusterTendency_channel_0_skewness",
            "pixel_original_glcm_ClusterTendency_channel_0_kurtosis",
            "pixel_original_glcm_ClusterTendency_channel_0_lognorm_fit_p0",
            "pixel_original_glcm_ClusterTendency_channel_0_lognorm_fit_p2",
        ]
    ) == set(df.columns)
