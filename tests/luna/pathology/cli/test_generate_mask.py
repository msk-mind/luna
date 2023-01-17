import os

import openslide
from click.testing import CliRunner

from luna.pathology.cli.generate_mask import cli


def test_cli(tmp_path):
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "tests/testdata/pathology/123.svs",
            "tests/testdata/pathology/test-project/pathology_annotations/123456_annotation_from_halo.xml",
            "-o",
            tmp_path,
            "-an",
            "Tumor",
        ],
    )

    assert result.exit_code == 0

    assert os.path.exists(f"{tmp_path}/mask_full_res.tif")
    assert os.path.exists(f"{tmp_path}/metadata.yml")

    openslide.ImageSlide(f"{tmp_path}/mask_full_res.tif")
