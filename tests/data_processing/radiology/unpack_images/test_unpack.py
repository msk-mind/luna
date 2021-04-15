import pytest
import os, shutil
from pyspark import SQLContext
from click.testing import CliRunner
import os

from data_processing.radiology.unpack_images.unpack import cli


unpacked_pngs_path = "tests/data_processing/radiology/testdata/unpacked_pngs"
app_config_path = unpacked_pngs_path + "/app_config.yaml"
data_config_path = unpacked_pngs_path + "/data_config.yaml"

def test_cli():

    runner = CliRunner()
    result = runner.invoke(cli, 
        ['-d', 'tests/data_processing/radiology/unpack_images/data.yaml',
        '-a', 'tests/test_config.yaml'])

    assert result.exit_code == 0
    assert os.path.exists(unpacked_pngs_path)
    assert os.path.exists(app_config_path)
    assert os.path.exists(data_config_path)

    if os.path.exists(unpacked_pngs_path):
        shutil.rmtree(unpacked_pngs_path)
