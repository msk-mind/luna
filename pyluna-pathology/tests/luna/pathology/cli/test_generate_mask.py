import os
from click.testing import CliRunner

from luna.pathology.cli.generate_mask import cli
import openslide


def test_cli(tmp_path):
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "pyluna-pathology/tests/luna/pathology/cli/testdata/data/123.svs",
            "pyluna-pathology/tests/luna/pathology/testdata/data/test-project/pathology_annotations/123456_annotation_from_halo.xml",
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
