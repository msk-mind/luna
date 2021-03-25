import pytest
import os, shutil
from pyspark import SQLContext
from click.testing import CliRunner
import os

from data_processing.common.config import ConfigSet
from data_processing.common.sparksession import SparkConfig
from data_processing.radiology.refined_table.annotation.generate import cli
import data_processing.common.constants as const

project_path = "tests/data_processing/radiology/testdata/test-project"
png_table_path = project_path + "/tables/PNG_dsn"
png_config_path = project_path + "/configs/PNG_dsn"
app_config_path = png_config_path + "/app_config.yaml"
data_config_path = png_config_path + "/data_config.yaml"

@pytest.fixture(autouse=True)
def spark():
    print('------setup------')
    ConfigSet(name=const.APP_CFG, config_file='tests/test_config.yaml')
    spark = SparkConfig().spark_session(config_name=const.APP_CFG, app_name='test-radiology-refined-annotation')

    yield spark

    print('------teardown------')
    if os.path.exists(png_table_path):
        shutil.rmtree(png_table_path)
    if os.path.exists(project_path+"/configs"):
        shutil.rmtree(project_path+"/configs")


def test_cli(spark):

    runner = CliRunner()
    result = runner.invoke(cli, 
        ['-d', 'tests/data_processing/radiology/refined_table/annotation/data.yaml',
        '-a', 'tests/test_config.yaml'])

    assert result.exit_code == 0

    assert os.path.exists(app_config_path)
    assert os.path.exists(data_config_path)

    df = spark.read.format("delta").load(png_table_path)
    df.show(10)
    assert df.count() == 1
    df.unpersist()
