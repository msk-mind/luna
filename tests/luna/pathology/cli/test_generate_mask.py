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
