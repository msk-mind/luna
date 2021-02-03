import pytest
import requests
from pytest_mock import mocker
import os, shutil
from pyspark import SQLContext
from click.testing import CliRunner
import os

from data_processing.common.config import ConfigSet
from data_processing.common.sparksession import SparkConfig
from data_processing.pathology.proxy_table.regional_annotation.generate import cli, convert_bmp_to_npy
import data_processing.common.constants as const
from tests.data_processing.pathology.proxy_table.regional_annotation.request_mock import CSVMockResponse

# proxy_table_path = "tests/data_processing/testdata/data/test-project/tables/BITMASK"
# landing_path = "tests/data_processing/testdata/data/test-project/wsi-project"

spark = None
LANDING_PATH = None

def setup_module(module):
    """ setup any state specific to the execution of the given module."""
    ConfigSet(name=const.APP_CFG, config_file='tests/test_config.yaml')
    ConfigSet(name=const.DATA_CFG, config_file='tests/data_processing/pathology/'
                                               'proxy_table/regional_annotation/data_config.yaml')
    module.spark = SparkConfig().spark_session(config_name=const.APP_CFG, app_name='test-pathology-annotation-proxy')

    cfg = ConfigSet()
    module.LANDING_PATH = cfg.get_value(path=const.DATA_CFG + '::LANDING_PATH')

    if os.path.exists(LANDING_PATH):
        shutil.rmtree(LANDING_PATH)
    os.makedirs(LANDING_PATH)


def teardown_module(module):
    """ teardown any state that was previously setup with a setup_module
    method.
    """
    shutil.rmtree(LANDING_PATH)


def test_convert_bmp_to_npy():
    actual_path = convert_bmp_to_npy('tests/data_processing/pathology/proxy_table/'
                                         'regional_annotation/test_data/input/24bpp-topdown-320x240.bmp',
                                         LANDING_PATH)

    expected_path = os.path.join(LANDING_PATH, 'input/24bpp-topdown-320x240.npy')
    assert actual_path == expected_path
    assert os.path.exists(expected_path)
        
def test_cli(monkeypatch):
    monkeypatch.setenv("MIND_GPFS_DIR", "")
    monkeypatch.setenv("HDFS_URI", "")

    # mock request to slideviewer api
    def mock_get(*args, **kwargs):
        return CSVMockResponse()

    monkeypatch.setattr(requests, "get", mock_get)

    runner = CliRunner()
    result = runner.invoke(cli, 
        ['-d', 'tests/data_processing/pathology/proxy_table/annotation/data_config.yaml',
        '-a', 'tests/test_config.yaml'])

    assert result.exit_code == 0

    # df = spark.read.format("delta").load(proxy_table_path)
    # assert df.count() == 1
    # df.unpersist()
