import os

import fire
import pandas as pd

from luna.pathology.cli.extract_shape_features import cli


def test_cli_generate_mask(tmp_path):
    fire.Fire(
        cli,
        [
            "--slide_mask_urlpath",
            "tests/testdata/pathology/generate_tile_mask//tile_mask.tif",
            "--output_urlpath",
            str(tmp_path),
            "--label_cols",
            "Background,Tumor",
        ],
    )

    assert os.path.exists(f"{tmp_path}/shape_features.csv")
    assert os.path.exists(f"{tmp_path}/metadata.yml")
    df = pd.read_csv(f"{tmp_path}/shape_features.csv")

    assert len(df) == 178
