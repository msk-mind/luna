import pytest
import requests
from pytest_mock import mocker
import os, shutil
from pyspark import SQLContext
from click.testing import CliRunner
import os

from data_processing.common.config import ConfigSet
from data_processing.common.sparksession import SparkConfig
from data_processing.pathology.proxy_table.annotation.generate import cli
import data_processing.common.constants as const
from tests.data_processing.pathology.proxy_table.annotation.request_mock import MockResponse

# proxy_table_path = "tests/data_processing/testdata/data/test-project/tables/BITMASK"
# landing_path = "tests/data_processing/testdata/data/test-project/wsi-project"

@pytest.fixture(autouse=True)
def spark(monkeypatch):
    print('------setup------')
    ConfigSet(name=const.APP_CFG, config_file='tests/test_config.yaml')
    spark = SparkConfig().spark_session(config_name=const.APP_CFG, app_name='test-pathology-annotation-proxy')
    monkeypatch.setenv("MIND_GPFS_DIR", "")
    monkeypatch.setenv("HDFS_URI", "")

    # mock request to slideviewer api
    def mock_get(*args, **kwargs):
        return MockResponse()

    monkeypatch.setattr(requests, "get", mock_get)

    yield spark

    print('------teardown------')
    # if os.path.exists(proxy_table_path):
    #     shutil.rmtree(proxy_table_path)
    # if os.path.exists(landing_path):
    #     shutil.rmtree(landing_path)
        
def test_cli(spark):

    runner = CliRunner()
    result = runner.invoke(cli, 
        ['-d', 'tests/data_processing/pathology/proxy_table/annotation/data_config.yaml',
        '-a', 'tests/test_config.yaml'])

    assert result.exit_code == 0

    # df = spark.read.format("delta").load(proxy_table_path)
    # assert df.count() == 1
    # df.unpersist()
