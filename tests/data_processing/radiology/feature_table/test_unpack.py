import pytest
import os, shutil
from pyspark import SQLContext
from click.testing import CliRunner
import os

from data_processing.radiology.feature_table.unpack import cli


unpacked_pngs_path = "tests/data_processing/testdata/data/unpacked_pngs"

def test_cli():

    runner = CliRunner()
    result = runner.invoke(cli, 
        ['-d', 'tests/data_processing/radiology/feature_table/data.yaml',
        '-a', 'tests/test_config.yaml'])

    assert result.exit_code == 0
    assert os.path.exists(unpacked_pngs_path)

    if os.path.exists(unpacked_pngs_path):
        shutil.rmtree(unpacked_pngs_path)
