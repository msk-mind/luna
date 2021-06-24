import pytest
import os, shutil
from click.testing import CliRunner

from data_processing.common.config import ConfigSet
from data_processing.common.sparksession import SparkConfig
from data_processing.pathology.point_annotation.refined_table.generate import cli
import data_processing.common.constants as const


point_geojson_table_path = "tests/data_processing/pathology/point_annotation/testdata/test-project/tables/POINT_GEOJSON_dsn"
config_path = "tests/data_processing/pathology/point_annotation/testdata/test-project/configs"
app_config_path = config_path + "/POINT_GEOJSON_dsn/app_config.yaml"
data_config_path = config_path + "/POINT_GEOJSON_dsn/data_config.yaml"

@pytest.fixture(autouse=True)
def spark():
    print('------setup------')
    ConfigSet(name=const.APP_CFG, config_file='tests/test_config.yaml')
    spark = SparkConfig().spark_session(config_name=const.APP_CFG, app_name='test-point-annot-refined')

    yield spark

    print('------teardown------')
    if os.path.exists(point_geojson_table_path):
        shutil.rmtree(point_geojson_table_path)
    if os.path.exists(config_path):
        shutil.rmtree(config_path)

def test_cli(spark):

    runner = CliRunner()
    result = runner.invoke(cli, 
        ['-d', 'tests/data_processing/pathology/point_annotation/testdata/point_geojson_config.yaml',
         '-a', 'tests/test_config.yaml'])

    print(result.exc_info)
    assert result.exit_code == 0

    assert os.path.exists(app_config_path)
    assert os.path.exists(data_config_path)

    df = spark.read.format("parquet").load(point_geojson_table_path)
    df.show(10, False)
    assert df.count() == 1
    df.unpersist()
