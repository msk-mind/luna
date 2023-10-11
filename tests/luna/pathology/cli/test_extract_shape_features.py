import os

import fire
import pandas as pd

from luna.pathology.cli.extract_shape_features import cli


def test_cli_generate_mask_s3(s3fs_client):
    s3fs_client.mkdirs("testmask", exist_ok=True)
    s3fs_client.put(
        "tests/testdata/pathology/generate_tile_mask/tile_mask.tif", "testmask/test/"
    )
    fire.Fire(
        cli,
        [
            "--slide_mask_urlpath",
            "s3://testmask/test/tile_mask.tif",
            "--output_urlpath",
            "s3://testmask/out",
            "--label_cols",
            "Background,Tumor",
            "--storage_options",
            "{'key': '', 'secret': '', 'client_kwargs': {'endpoint_url': '"
            + s3fs_client.client_kwargs["endpoint_url"]
            + "'}}",
        ],
    )

    assert s3fs_client.exists("testmask/out/shape_features.csv")


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
