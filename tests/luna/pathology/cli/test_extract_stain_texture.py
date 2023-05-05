import os

import fire
import pandas as pd

from luna.pathology.cli.extract_stain_texture import cli


def test_cli(tmp_path):
    fire.Fire(
        cli,
        [
            "--slide_image_urlpath",
            "tests/testdata/pathology/123.svs",
            "--slide_mask_urlpath",
            "tests/testdata/pathology/generate_mask/mask_full_res.tif",
            "--output_urlpath",
            str(tmp_path),
            "--tile_size",
            str(512),
            "--stain_channel",
            str(0),
            "--stain_sample_factor",
            str(10),
        ],
    )

    assert os.path.exists(f"{tmp_path}/stainomics.parquet")

    df = pd.read_parquet(f"{tmp_path}/stainomics.parquet")

    assert bool(df.notnull().values.any()) is True
    assert df.shape == (1, 216)
