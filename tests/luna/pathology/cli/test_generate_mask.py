import os

import fire
import openslide

from luna.pathology.cli.generate_mask import cli


def test_cli(tmp_path):
    fire.Fire(
        cli,
        [
            "--slide-urlpath",
            "tests/testdata/pathology/123.svs",
            "--roi-urlpath",
            "tests/testdata/pathology/test-project/pathology_annotations/123456_annotation_from_halo.xml",
            "--output-urlpath",
            str(tmp_path),
            "--annotation-name",
            "Tumor",
        ],
    )

    assert os.path.exists(f"{tmp_path}/mask_full_res.tif")
    assert os.path.exists(f"{tmp_path}/metadata.yml")

    openslide.ImageSlide(f"{tmp_path}/mask_full_res.tif")


def test_cli_s3(s3fs_client):
    s3fs_client.mkdirs("masktest", exist_ok=True)
    s3fs_client.put("tests/testdata/pathology/123.svs", "masktest/test/")
    s3fs_client.put(
        "tests/testdata/pathology/test-project/pathology_annotations/123456_annotation_from_halo.xml",
        "masktest/test/",
    )
    fire.Fire(
        cli,
        [
            "--slide-urlpath",
            "s3://masktest/test/123.svs",
            "--roi-urlpath",
            "s3://masktest/test/123456_annotation_from_halo.xml",
            "--output-urlpath",
            "s3://masktest/out/",
            "--annotation-name",
            "Tumor",
            "--storage_options",
            "{'key': '', 'secret': '', 'client_kwargs': {'endpoint_url': '"
            + s3fs_client.client_kwargs["endpoint_url"]
            + "'}}",
        ],
    )

    assert s3fs_client.exists("s3://masktest/out/mask_full_res.tif")
    assert s3fs_client.exists("s3://masktest/out/metadata.yml")
