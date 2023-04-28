import fire

from luna.radiology.cli.dicom_to_itk import dicom_to_itk

data_dir = ""


def test_cli_nii(tmp_path):
    fire.Fire(
        dicom_to_itk,
        [
            "--dicom-urlpath",
            "tests/testdata/radiology/2.000000-CTAC-24716/dicoms/",
            "--output-urlpath",
            str(tmp_path),
            "--itk_c_type",
            "float",
            "--itk_image_type",
            "nii",
        ],
    )

    # assert os.path.exists(str(tmp_path) + "/metadata.yml")

    # with open((str(tmp_path) + "/metadata.yml"), "r") as fp:
    # metadata = yaml.safe_load(fp)

    # assert os.path.exists(metadata["itk_volume"])

    # assert Path(metadata["itk_volume"]).suffix == ".nii"


def test_cli_mhd(tmp_path):
    fire.Fire(
        dicom_to_itk,
        [
            "--dicom-urlpath",
            "tests/testdata/radiology/2.000000-CTAC-24716/dicoms/",
            "--output-urlpath",
            str(tmp_path),
            "--itk_c_type",
            "float",
            "--itk_image_type",
            "mhd",
        ],
    )

    # assert os.path.exists(str(tmp_path) + "/metadata.yml")

    # with open((str(tmp_path) + "/metadata.yml"), "r") as fp:
    # metadata = yaml.safe_load(fp)

    # assert os.path.exists(metadata["itk_volume"])

    # assert Path(metadata["itk_volume"]).suffix == ".mhd"
