import os

import fire
import numpy as np
import tifffile

from luna.pathology.cli.generate_tile_mask import cli


def test_cli_generate_mask(tmp_path):
    fire.Fire(
        cli,
        [
            "--slide_urlpath",
            "tests/testdata/pathology/123.svs",
            "--tiles_urlpath",
            "tests/testdata/pathology/infer_tumor_background/123/tile_scores_and_labels_pytorch_inference.parquet",
            "--output-urlpath",
            str(tmp_path),
            "--label_cols",
            "Background,Tumor",
        ],
    )

    assert os.path.exists(f"{tmp_path}/tile_mask.tif")
    assert os.path.exists(f"{tmp_path}/metadata.yml")

    mask = tifffile.imread(f"{tmp_path}/tile_mask.tif")

    assert np.array_equal(np.unique(mask), [0, 2])
