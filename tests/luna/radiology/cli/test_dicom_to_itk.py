import os
from pathlib import Path

import yaml
from click.testing import CliRunner

from luna.radiology.cli.dicom_to_itk import cli

data_dir = ""


def test_cli_nii(tmp_path):

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "tests/testdata/radiology/2.000000-CTAC-24716/dicoms/",
            "-o",
            tmp_path,
            "--itk_c_type",
            "float",
            "--itk_image_type",
            "nii",
        ],
    )

    assert result.exit_code == 0
    assert os.path.exists(str(tmp_path) + "/metadata.yml")

    with open((str(tmp_path) + "/metadata.yml"), "r") as fp:
        metadata = yaml.safe_load(fp)

    assert os.path.exists(metadata["itk_volume"])

    assert Path(metadata["itk_volume"]).suffix == ".nii"


def test_cli_mhd(tmp_path):

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "tests/testdata/radiology/2.000000-CTAC-24716/dicoms/",
            "-o",
            tmp_path,
            "--itk_c_type",
            "float",
            "--itk_image_type",
            "mhd",
        ],
    )

    assert result.exit_code == 0
    assert os.path.exists(str(tmp_path) + "/metadata.yml")

    with open((str(tmp_path) + "/metadata.yml"), "r") as fp:
        metadata = yaml.safe_load(fp)

    assert os.path.exists(metadata["itk_volume"])

    assert Path(metadata["itk_volume"]).suffix == ".mhd"