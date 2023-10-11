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


def test_cli_s3(s3fs_client):
    s3fs_client.mkdirs("teststain", exist_ok=True)
    s3fs_client.put("tests/testdata/pathology/123.svs", "teststain/test/")
    s3fs_client.put(
        "tests/testdata/pathology/generate_mask/mask_full_res.tif", "teststain/test/"
    )

    fire.Fire(
        cli,
        [
            "--slide_image_urlpath",
            "s3://teststain/test/123.svs",
            "--slide_mask_urlpath",
            "s3://teststain/test/mask_full_res.tif",
            "--output_urlpath",
            "s3://teststain/out",
            "--tile_size",
            str(512),
            "--stain_channel",
            str(0),
            "--stain_sample_factor",
            str(10),
            "--storage_options",
            "{'key': '', 'secret': '', 'client_kwargs': {'endpoint_url': '"
            + s3fs_client.client_kwargs["endpoint_url"]
            + "'}}",
        ],
    )

    assert s3fs_client.exists("teststain/out/stainomics.parquet")
