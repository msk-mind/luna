from pathlib import Path

import fire
import numpy as np
import pandas as pd

from luna.radiology.cli.extract_radiomics import extract_radiomics_multiple_labels


def test_cli_extract_radiomics(tmp_path):
    fire.Fire(
        extract_radiomics_multiple_labels,
        [
            "tests/testdata/radiology/2.000000-CTAC-24716/volumes/image.mhd",
            "tests/testdata/radiology/2.000000-CTAC-24716/volumes/label.mha",
            "--lesion-indices",
            "1",
            "--pyradiomics_config",
            '{"interpolator": "sitkBSpline","resampledPixelSpacing":[2.5,2.5,2.5]}',
            "--output-urlpath",
            str(tmp_path),
        ],
    )

    # assert os.path.exists(str(tmp_path) + "/metadata.yml")

    # with open((str(tmp_path) + "/metadata.yml"), "r") as fp:
    # metadata = yaml.safe_load(fp)

    # assert os.path.exists(metadata["feature_csv"])

    df = pd.read_csv(Path(tmp_path) / "radiomics.csv")
    assert np.allclose(df.loc[0, "original_ngtdm_Strength"], 7.806095657492503)
    assert np.allclose(df.loc[0, "original_shape_MajorAxisLength"], 75.83532054333504)
    assert df.loc[0, "lesion_index"] == 1
