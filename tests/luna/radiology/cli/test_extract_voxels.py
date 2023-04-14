import os

import numpy as np
import yaml
from click.testing import CliRunner

from luna.radiology.cli.extract_voxels import cli


def test_cli_voxels(tmp_path):

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "tests/testdata/radiology/2.000000-CTAC-24716/volumes/image.mhd",
            "-o",
            tmp_path,
        ],
    )

    assert result.exit_code == 0
    assert os.path.exists(str(tmp_path) + "/metadata.yml")

    with open((str(tmp_path) + "/metadata.yml"), "r") as fp:
        metadata = yaml.safe_load(fp)

    assert os.path.exists(metadata["npy_volume"])

    arr = np.load(metadata["npy_volume"])

    assert arr.shape == (512, 512, 9)
    assert np.allclose(np.mean(arr), -1290.3630044725205)