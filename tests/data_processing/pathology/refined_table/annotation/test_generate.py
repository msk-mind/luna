import pytest
import os, shutil
from click.testing import CliRunner
import os

from data_processing.common.config import ConfigSet
from data_processing.common.sparksession import SparkConfig
from data_processing.pathology.refined_table.annotation.generate import cli
import data_processing.common.constants as const
from data_processing.common.Neo4jConnection import Neo4jConnection


geojson_table_path = "tests/data_processing/testdata/data/test-project/tables/GEOJSON_dsn"
concat_geojson_table_path = "tests/data_processing/testdata/data/test-project/tables/CONCAT_GEOJSON_ds"
concat_geojson_data_path = "tests/data_processing/testdata/data/test-project/pathology_annotations/regional_concat_geojsons"

@pytest.fixture(autouse=True)
def spark():
    print('------setup------')
    ConfigSet(name=const.APP_CFG, config_file='tests/test_config.yaml')
    spark = SparkConfig().spark_session(config_name=const.APP_CFG, app_name='test-pathology-refined-annot')

    yield spark

    print('------teardown------')
    clean_up_paths = [geojson_table_path, concat_geojson_data_path, concat_geojson_table_path]
    for path in clean_up_paths:
        if os.path.exists(path):
            shutil.rmtree(path)
        
def test_cli_geojson(mocker, spark):

    # mock graph connection
    mocker.patch.object(Neo4jConnection, 'query')

    runner = CliRunner()
    result = runner.invoke(cli, 
        ['-t', 'tests/data_processing/pathology/refined_table/annotation/geojson_data.yaml',
         '-l', 'tests/data_processing/pathology/test_regional_etl_config.yaml',
         '-f', 'tests/test_config.yaml',
         '-p', 'geojson'])

    assert result.exit_code == 0

    df = spark.read.format("delta").load(geojson_table_path)
    df.show(10, False)
    assert df.count() == 2
    df.unpersist()


def test_cli_concat(mocker, spark):

    # mock graph connection
    mocker.patch.object(Neo4jConnection, 'query')

    runner = CliRunner()
    result = runner.invoke(cli,
                           ['-t', 'tests/data_processing/pathology/refined_table/annotation/geojson_concat_data.yaml',
                            '-l', 'tests/data_processing/pathology/test_regional_etl_config.yaml',
                            '-f', 'tests/test_config.yaml',
                            '-p', 'concat'])

    assert result.exit_code == 0

    df = spark.read.format("delta").load(concat_geojson_table_path)
    df.show(10, False)
    assert df.count() == 1
    df.unpersist()