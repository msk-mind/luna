import pytest
from pytest_mock import mocker
import os, shutil
from pyspark import SQLContext
from click.testing import CliRunner
import os

from data_processing.common.config import ConfigSet
from data_processing.common.sparksession import SparkConfig
from data_processing.pathology.proxy_table.generate import cli
import data_processing.common.constants as const


proxy_table_path = "tests/data_processing/testdata/data/test-project/tables/WSI_dsn"
landing_path = "tests/data_processing/testdata/data/test-project/wsi"

@pytest.fixture(autouse=True)
def spark(monkeypatch):
    print('------setup------')
    ConfigSet(name=const.APP_CFG, config_file='tests/test_config.yaml')
    spark = SparkConfig().spark_session(config_name=const.APP_CFG, app_name='test-pathology-proxy')
    monkeypatch.setenv("MIND_GPFS_DIR", "")
    monkeypatch.setenv("HDFS_URI", "")

    yield spark

    print('------teardown------')
    if os.path.exists(proxy_table_path):
        shutil.rmtree(proxy_table_path)
    if os.path.exists(landing_path):
        shutil.rmtree(landing_path)
        
def test_cli(spark):

    runner = CliRunner()
    result = runner.invoke(cli, 
        ['-t', 'tests/data_processing/pathology/proxy_table/data.yaml',
        '-f', 'tests/test_config.yaml'])

    assert result.exit_code == 0

    df = spark.read.format("delta").load(proxy_table_path)
    assert df.count() == 1
    df.unpersist()
