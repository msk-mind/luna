import pytest
import os, shutil
from click.testing import CliRunner

from data_processing.common.config import ConfigSet
from data_processing.common.sparksession import SparkConfig
from data_processing.pathology.point_annotation.refined_table.generate import cli
import data_processing.common.constants as const


point_geojson_table_path = "tests/data_processing/testdata/data/test-project/tables/POINT_GEOJSON_ds"

@pytest.fixture(autouse=True)
def spark():
    print('------setup------')
    ConfigSet(name=const.APP_CFG, config_file='tests/test_config.yaml')
    spark = SparkConfig().spark_session(config_name=const.APP_CFG, app_name='test-point-annot-refined')

    yield spark

    print('------teardown------')
    if os.path.exists(point_geojson_table_path):
        shutil.rmtree(point_geojson_table_path)
        
def test_cli(spark):

    runner = CliRunner()
    result = runner.invoke(cli, 
        ['-t', 'tests/data_processing/pathology/point_annotation/testdata/point_geojson_config.yaml',
         '-f', 'tests/test_config.yaml'])

    assert result.exit_code == 0

    df = spark.read.format("delta").load(point_geojson_table_path)
    df.show(10, False)
    assert df.count() == 2
    df.unpersist()
