from pathlib import Path

import fire
import numpy as np

from luna.radiology.cli.extract_voxels import extract_voxels


def test_cli_voxels(tmp_path):
    fire.Fire(
        extract_voxels,
        [
            "tests/testdata/radiology/2.000000-CTAC-24716/volumes/image.mhd",
            "--output-dir",
            str(tmp_path),
        ],
    )

    # assert os.path.exists(str(tmp_path) + "/metadata.yml")

    # with open((str(tmp_path) + "/metadata.yml"), "r") as fp:
    # metadata = yaml.safe_load(fp)

    # assert os.path.exists(metadata["npy_volume"])

    arr = np.load(Path(tmp_path) / "image.npy")

    assert arr.shape == (512, 512, 9)
    assert np.allclose(np.mean(arr), -1290.3630044725205)
