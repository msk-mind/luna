import os

import fire
import pandas as pd

from luna.pathology.cli.extract_tile_shape_features import cli


def test_cli_extract_tile_shape_features(tmp_path):
    fire.Fire(
        cli,
        [
            "--slide_urlpath",
            "tests/testdata/pathology/123.svs",
            "--object_urlpath",
            "tests/testdata/pathology/test_cell_detections.geojson",
            "--tiles_urlpath",
            "tests/testdata/pathology/infer_tumor_background/123/tile_scores_and_labels_pytorch_inference.parquet",
            "--output_urlpath",
            str(tmp_path),
            "--label_cols",
            "Background,Tumor",
        ],
    )

    assert os.path.exists(f"{tmp_path}/shape_features.parquet")
    assert os.path.exists(f"{tmp_path}/metadata.yml")
    df = pd.read_parquet(f"{tmp_path}/shape_features.parquet")

    assert len(df) == 866


def test_cli_extract_tile_shape_features_s3(s3fs_client):
    s3fs_client.mkdirs("testtile", exist_ok=True)
    s3fs_client.put("tests/testdata/pathology/123.svs", "testtile/test/")
    s3fs_client.put(
        "tests/testdata/pathology/test_cell_detections.geojson", "testtile/test/"
    )
    s3fs_client.put(
        "tests/testdata/pathology/infer_tumor_background/123/tile_scores_and_labels_pytorch_inference.parquet",
        "testtile/test/",
    )
    fire.Fire(
        cli,
        [
            "--slide_urlpath",
            "s3://testtile/test/123.svs",
            "--object_urlpath",
            "s3://testtile/test/test_cell_detections.geojson",
            "--tiles_urlpath",
            "s3://testtile/test/tile_scores_and_labels_pytorch_inference.parquet",
            "--output_urlpath",
            "s3://testtile/out/",
            "--label_cols",
            "Background,Tumor",
            "--storage_options",
            "{'key': '', 'secret': '', 'client_kwargs': {'endpoint_url': '"
            + s3fs_client.client_kwargs["endpoint_url"]
            + "'}}",
        ],
    )

    assert s3fs_client.exists("s3://testtile/out/shape_features.parquet")
    assert s3fs_client.exists("s3://testtile/out/metadata.yml")
