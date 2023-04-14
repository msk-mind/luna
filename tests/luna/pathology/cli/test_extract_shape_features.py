import os

import pandas as pd
from click.testing import CliRunner

from luna.pathology.cli.extract_shape_features import cli


def test_cli_generate_mask(tmp_path):
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "tests/testdata/pathology/generate_tile_mask//tile_mask.tif",
            "-o",
            tmp_path,
            "-lc",
            "Background,Tumor",
        ],
    )

    assert result.exit_code == 0

    assert os.path.exists(f"{tmp_path}/shape_features.csv")
    assert os.path.exists(f"{tmp_path}/metadata.yml")
    df = pd.read_csv(f"{tmp_path}/shape_features.csv")

    assert len(df) == 2