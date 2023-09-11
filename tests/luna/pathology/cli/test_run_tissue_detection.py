import os

import fire

from luna.pathology.cli.run_tissue_detection import cli


def test_otsu_s3(dask_client, s3fs_client):
    s3fs_client.mkdirs("tissuetest", exist_ok=True)
    s3fs_client.put("tests/testdata/pathology/123.svs", "tissuetest/test/")
    fire.Fire(
        cli,
        [
            "--slide-urlpath",
            "s3://tissuetest/test/123.svs",
            "--tile-size",
            str(256),
            "--batch-size",
            str(8),
            "--output-urlpath",
            "s3://tissuetest/test",
            "--filter-query",
            "otsu_score > 0.5",
            "--storage_options",
            "{'key': '', 'secret': '', 'client_kwargs': {'endpoint_url': '"
            + s3fs_client.client_kwargs["endpoint_url"]
            + "'}}",
        ],
    )
    assert s3fs_client.exists("tissuetest/test/123.tiles.parquet")
    assert s3fs_client.exists("tissuetest/test/metadata.yml")


def test_otsu(tmp_path, dask_client):
    fire.Fire(
        cli,
        [
            "--slide-urlpath",
            "tests/testdata/pathology/123.svs",
            "--tile-size",
            str(256),
            "--batch-size",
            str(8),
            "--output-urlpath",
            str(tmp_path),
            "--filter-query",
            "otsu_score > 0.5",
        ],
    )

    assert os.path.exists(f"{tmp_path}/123.tiles.parquet")
    assert os.path.exists(f"{tmp_path}/metadata.yml")


def test_stain(tmp_path, dask_client):
    fire.Fire(
        cli,
        [
            "--slide-urlpath",
            "tests/testdata/pathology/123.svs",
            "--output-urlpath",
            str(tmp_path),
            "--thumbnail-magnification",
            str(5),
            "--tile-size",
            str(256),
            "--batch-size",
            str(8),
            "--filter-query",
            "stain0_score > 0.05",
        ],
    )

    assert os.path.exists(f"{tmp_path}/123.tiles.parquet")
    assert os.path.exists(f"{tmp_path}/metadata.yml")
