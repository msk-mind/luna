import pytest
import os, shutil
from click.testing import CliRunner

from data_processing.common.config import ConfigSet
from data_processing.common.sparksession import SparkConfig
from data_processing.pathology.point_annotation.proxy_table.generate import cli
import data_processing.common.constants as const


point_json_table_path = "tests/data_processing/pathology/point_annotation/testdata/test-project/tables/POINT_RAW_JSON_ds"

@pytest.fixture(autouse=True)
def spark():
    print('------setup------')
    ConfigSet(name=const.APP_CFG, config_file='tests/test_config.yaml')
    spark = SparkConfig().spark_session(config_name=const.APP_CFG, app_name='test-point-annot-proxy')

    yield spark

    print('------teardown------')
    if os.path.exists(point_json_table_path):
        shutil.rmtree(point_json_table_path)
        
def test_cli(spark):

    runner = CliRunner()
    result = runner.invoke(cli, 
        ['-d', 'tests/data_processing/pathology/point_annotation/testdata/point_js_config.yaml',
         '-a', 'tests/test_config.yaml'])

    assert result.exit_code == 0

    df = spark.read.format("delta").load(point_json_table_path)
    df.show(10, False)
    assert df.count() == 2
    df.unpersist()
