import os

import numpy as np
import tifffile
from click.testing import CliRunner

from luna.pathology.cli.generate_tile_mask import cli


def test_cli_generate_mask(tmp_path):
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "tests/testdata/pathology/123.svs",
            "tests/testdata/pathology/infer_tumor_background/123/",
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
