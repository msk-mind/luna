import pytest
import os, shutil
from pyspark import SQLContext
from click.testing import CliRunner
import os

from data_processing.radiology.feature_table.unpack import cli


unpacked_pngs_path = "tests/data_processing/testdata/data/unpacked_pngs"
config_path = "tests/data_processing/testdata/data/test-project/config"
app_config_path = config_path + "/FEATURE_unpack/app_config.yaml"
data_config_path = config_path + "/FEATURE_unpack/data_config.yaml"

def test_cli():

    runner = CliRunner()
    result = runner.invoke(cli, 
        ['-d', 'tests/data_processing/radiology/feature_table/data.yaml',
        '-a', 'tests/test_config.yaml'])

    assert result.exit_code == 0
    assert os.path.exists(unpacked_pngs_path)
    assert os.path.exists(app_config_path)
    assert os.path.exists(data_config_path)

    if os.path.exists(unpacked_pngs_path):
        shutil.rmtree(unpacked_pngs_path)

    if os.path.exists(config_path):
        shutil.rmtree(config_path)
