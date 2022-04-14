import os
from click.testing import CliRunner

from luna.pathology.cli.generate_tile_labels import cli
import pandas as pd

def test_cli(tmp_path):
    runner = CliRunner()

    result = runner.invoke(
        cli,
        [
            "pyluna-pathology/tests/luna/pathology/cli/testdata/data/dsa_annots/",
            "pyluna-pathology/tests/luna/pathology/cli/testdata/data/save_tiles/123/",
            "-o",
            tmp_path,
        ],
    )

    assert result.exit_code == 0

    out_tile = pd.read_parquet(f"{tmp_path}/123.regional_label.tiles.parquet").reset_index().set_index('address')

    assert out_tile.loc["x1_y1_z10.0", "regional_label"] == "Other"
    assert out_tile.loc["x3_y4_z10.0", "regional_label"] == "Tumor"
