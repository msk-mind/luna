import os

import pandas as pd
from click.testing import CliRunner

from luna.pathology.cli.extract_stain_texture import cli


def test_cli(tmp_path):
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "tests/luna/pathology/cli/testdata/data/123.svs",
            "tests/luna/pathology/cli/testdata/data/generate_mask/mask_full_res.tif",
            "-o",
            tmp_path,
            "-tx",
            500,
            "-sc",
            0,
            "-sf",
            10,
        ],
    )

    assert result.exit_code == 0
    assert os.path.exists(f"{tmp_path}/stainomics.parquet")

    df = pd.read_parquet(f"{tmp_path}/stainomics.parquet")

    assert bool(df.notnull().values.any()) is True
    assert df.shape == (1, 216)
