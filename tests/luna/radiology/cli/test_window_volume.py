from pathlib import Path

import fire
import medpy.io
import numpy as np

from luna.radiology.cli.window_volume import window_volume


def test_cli_window(tmp_path):
    fire.Fire(
        window_volume,
        [
            "tests/testdata/radiology/2.000000-CTAC-24716/volumes/image.mhd",
            "--output_dir",
            str(tmp_path),
            "--low_level",
            "0",
            "--high_level",
            "100",
        ],
    )

    # assert os.path.exists(str(tmp_path) + "/metadata.yml")

    # with open((str(tmp_path) + "/metadata.yml"), "r") as fp:
    # metadata = yaml.safe_load(fp)

    image, _ = medpy.io.load(Path(tmp_path) / "image.windowed.mhd")

    assert np.max(image) == 100
    assert np.min(image) == 0
