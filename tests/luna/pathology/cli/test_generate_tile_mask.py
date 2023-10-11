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

    assert np.array_equal(np.unique(mask), [0, 1])


def test_cli_generate_mask_s3(s3fs_client):
    s3fs_client.mkdirs("tilemask", exist_ok=True)
    s3fs_client.put("tests/testdata/pathology/123.svs", "tilemask/test/")
    s3fs_client.put(
        "tests/testdata/pathology/infer_tumor_background/123/tile_scores_and_labels_pytorch_inference.parquet",
        "tilemask/test/",
    )
    fire.Fire(
        cli,
        [
            "--slide_urlpath",
            "s3://tilemask/test/123.svs",
            "--tiles_urlpath",
            "s3://tilemask/test/tile_scores_and_labels_pytorch_inference.parquet",
            "--output-urlpath",
            "s3://tilemask/out/",
            "--label_cols",
            "Background,Tumor",
            "--storage_options",
            "{'key': '', 'secret': '', 'client_kwargs': {'endpoint_url': '"
            + s3fs_client.client_kwargs["endpoint_url"]
            + "'}}",
        ],
    )

    assert s3fs_client.exists("s3://tilemask/out/tile_mask.tif")
    assert s3fs_client.exists("s3://tilemask/out/metadata.yml")
