import os, yaml
from pathlib import Path
from click.testing import CliRunner

from luna.radiology.cli.window_volume import cli 

import medpy.io
import numpy as np

def test_cli_window(tmp_path):

    runner = CliRunner()
    result = runner.invoke(cli, [
        'pyluna-radiology/tests/luna/testdata/data/2.000000-CTAC-24716/volumes/image.mhd',
        '-o', tmp_path,
        '--low_level', 0,
        '--high_level', 100])

    assert result.exit_code == 0
    assert os.path.exists(str(tmp_path) + '/metadata.yml')

    with open ((str(tmp_path) + '/metadata.yml'), 'r') as fp:
        metadata = yaml.safe_load(fp)

    assert os.path.exists(metadata['itk_volume'])

    image, _ = medpy.io.load(metadata['itk_volume'])

    assert np.max(image)==100
    assert np.min(image)==0

