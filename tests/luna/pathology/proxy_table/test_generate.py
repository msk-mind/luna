import pytest
import os, shutil
from click.testing import CliRunner
import os

from luna.common.config import ConfigSet
from luna.common.sparksession import SparkConfig
from luna.pathology.proxy_table.generate import cli
import luna.common.constants as const


proxy_table_path = "tests/luna/testdata/data/test-project/tables/WSI_dsn"
landing_path = "tests/luna/testdata/data"
app_config_path = landing_path + '/test-project/configs/WSI_dsn/app_config.yaml'
data_config_path = landing_path + '/test-project/configs/WSI_dsn/data_config.yaml'

@pytest.fixture(autouse=True)
def spark(monkeypatch):
    print('------setup------')
    ConfigSet(name=const.APP_CFG, config_file='tests/test_config.yml')
    spark = SparkConfig().spark_session(config_name=const.APP_CFG, app_name='test-pathology-proxy')
    monkeypatch.setenv("MIND_GPFS_DIR", "")
    monkeypatch.setenv("HDFS_URI", "")

    yield spark

    print('------teardown------')
    if os.path.exists(proxy_table_path):
        shutil.rmtree(proxy_table_path)
        
def test_cli(spark):

    runner = CliRunner()
    result = runner.invoke(cli, 
        ['-d', 'tests/luna/pathology/proxy_table/data.yaml',
        '-a', 'tests/test_config.yml',
        '-p', 'delta'])

    assert result.exit_code == 0

    assert os.path.exists(app_config_path)
    assert os.path.exists(data_config_path)

    df = spark.read.format("delta").load(proxy_table_path)
    assert df.count() == 1
    df.unpersist()
