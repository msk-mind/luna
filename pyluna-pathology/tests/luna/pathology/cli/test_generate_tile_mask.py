import os
from click.testing import CliRunner

from luna.pathology.cli.generate_tile_mask import cli
import tifffile
import numpy as np


def test_cli_generate_mask(tmp_path):
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "pyluna-pathology/tests/luna/pathology/cli/testdata/data/123.svs",
            "pyluna-pathology/tests/luna/pathology/cli/testdata/data/infer_tumor_background/123/",
            "-o",
            tmp_path,
            "-lc",
            "Background,Tumor",
        ],
    )

    assert result.exit_code == 0

    assert os.path.exists(f"{tmp_path}/tile_mask.tif")
    assert os.path.exists(f"{tmp_path}/metadata.yml")

    mask = tifffile.imread(f"{tmp_path}/tile_mask.tif")

    assert np.array_equal(np.unique(mask), [0, 2])
