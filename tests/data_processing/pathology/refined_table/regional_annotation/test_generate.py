import pytest
import os, shutil
from click.testing import CliRunner

from data_processing.common.config import ConfigSet
from data_processing.common.sparksession import SparkConfig
from data_processing.pathology.refined_table.regional_annotation.generate import cli
import data_processing.common.constants as const

project_path = "tests/data_processing/testdata/data/test-project"
geojson_table_path = project_path + "/tables/REGIONAL_GEOJSON_dsn"
geojson_app_config_path = project_path +  "/configs/REGIONAL_GEOJSON_dsn/app_config.yaml"
geojson_data_config_path = project_path + "/configs/REGIONAL_GEOJSON_dsn/data_config.yaml"

concat_geojson_table_path = project_path +  "/tables/REGIONAL_CONCAT_GEOJSON_ds"
concat_geojson_app_config_path = project_path +  "/configs/REGIONAL_CONCAT_GEOJSON_ds/app_config.yaml"
concat_geojson_data_config_path = project_path + "/configs/REGIONAL_CONCAT_GEOJSON_ds/data_config.yaml"

@pytest.fixture(autouse=True)
def spark():
    print('------setup------')
    ConfigSet(name=const.APP_CFG, config_file='tests/test_config.yaml')
    spark = SparkConfig().spark_session(config_name=const.APP_CFG, app_name='test-pathology-refined-annot')

    yield spark

    print('------teardown------')
    clean_up_paths = [geojson_table_path, concat_geojson_table_path]
    for path in clean_up_paths:
        if os.path.exists(path):
            shutil.rmtree(path)
        
def test_cli_geojson(spark):

    runner = CliRunner()
    result = runner.invoke(cli, 
        ['-d', 'tests/data_processing/pathology/refined_table/regional_annotation/geojson_data.yaml',
         '-a', 'tests/test_config.yaml',
         '-p', 'geojson'])

    assert result.exit_code == 0

    assert os.path.exists(geojson_app_config_path)
    assert os.path.exists(geojson_data_config_path)

    df = spark.read.format("delta").load(geojson_table_path)
    df.show(10, False)
    assert df.count() == 2
    df.unpersist()


def test_cli_concat(spark):

    runner = CliRunner()
    result = runner.invoke(cli,
                           ['-d', 'tests/data_processing/pathology/refined_table/regional_annotation/geojson_concat_data.yaml',
                            '-a', 'tests/test_config.yaml',
                            '-p', 'concat'])

    assert result.exit_code == 0

    assert os.path.exists(concat_geojson_app_config_path)
    assert os.path.exists(concat_geojson_data_config_path)

    df = spark.read.format("delta").load(concat_geojson_table_path)
    df.show(10, False)
    assert df.count() == 1
    df.unpersist()
