import os, json
from pathlib import Path
from click.testing import CliRunner

from luna.radiology.cli.extract_radiomics import cli 

import medpy.io
import numpy as np
import pandas as pd

def test_cli_extract_radiomics(tmp_path):

    runner = CliRunner()
    result = runner.invoke(cli, [
        'pyluna-radiology/tests/luna/testdata/data/2.000000-CTAC-24716/volumes/image.mhd',
        'pyluna-radiology/tests/luna/testdata/data/2.000000-CTAC-24716/volumes/label.mha',
        '-idx', '1', 
        '--pyradiomics_config', '{"interpolator": "sitkBSpline","resampledPixelSpacing":[2.5,2.5,2.5]}',
        '-o', tmp_path])

    assert result.exit_code == 0
    assert os.path.exists(str(tmp_path) + '/metadata.json')

    with open ((str(tmp_path) + '/metadata.json'), 'r') as fp:
        metadata = json.load(fp)

    assert os.path.exists(metadata['feature_csv'])

    df = pd.read_csv(metadata['feature_csv'])
    assert np.allclose (df.loc[0, 'original_ngtdm_Strength'], 7.806095657492503)
    assert np.allclose (df.loc[0, 'original_shape_MajorAxisLength'], 75.83532054333504)
    assert df.loc[0, 'lesion_index'] == 1

