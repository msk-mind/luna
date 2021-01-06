import pytest
from pytest_mock import mocker
import os, shutil
from pyspark import SQLContext
from click.testing import CliRunner
import os

from data_processing.common.config import ConfigSet
from data_processing.common.sparksession import SparkConfig
from data_processing.radiology.refined_table.annotation.generate import cli
import data_processing.common.constants as const


png_table_path = "tests/data_processing/testdata/data/test-project/tables/PNG_ds"

@pytest.fixture(autouse=True)
def spark():
    print('------setup------')
    ConfigSet(name=const.APP_CFG, config_file='tests/test_config.yaml')
    spark = SparkConfig().spark_session(config_name=const.APP_CFG, app_name='test-radiology-refined-annotation')

    yield spark

    print('------teardown------')
    if os.path.exists(png_table_path):
        shutil.rmtree(png_table_path)


def test_cli(spark):

    runner = CliRunner()
    result = runner.invoke(cli, 
        ['-t', 'tests/data_processing/radiology/refined_table/annotation/data.yaml',
        '-f', 'tests/test_config.yaml'])

    assert result.exit_code == 0

    df = spark.read.format("delta").load(png_table_path)
    assert df.count() == 1
    df.unpersist()
