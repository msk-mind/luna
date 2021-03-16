import os
import shutil

import pytest
from click.testing import CliRunner
from data_processing.radiology.proxy_table.generate import *
from data_processing.common.sparksession import SparkConfig
import data_processing.common.constants as const

landing_path = "tests/data_processing/radiology/proxy_table/test_data/OV_16-158/"
table_path = "tests/data_processing/radiology/proxy_table/test_data/OV_16-158/tables/CT_OV_16-158_CT_20201028"
test_ingestion_template = "tests/data_processing/radiology/proxy_table/test_data/OV_16-158/config/data_config.yaml"

@pytest.fixture(autouse=True)
def spark():
    print('------setup------')
    APP_CFG = 'APP_CFG'
    ConfigSet(name=APP_CFG, config_file='tests/test_config.yaml')
    spark = SparkConfig().spark_session(config_name=APP_CFG, app_name='test-radiology-proxy')

    yield spark

    print('------teardown------')
    if os.path.exists(landing_path):
        shutil.rmtree(landing_path)


def test_cli(spark):

    runner = CliRunner()
    result = runner.invoke(cli, 
        ['-d', 'tests/data_processing/data_ingestion_template_valid.yml',
        '-a', 'tests/test_config.yaml',
        '-p', 'delta'])

    assert result.exit_code == 0

    df = spark.read.format("delta").load(table_path)
    df.show()
    assert df.count() == 1
    assert "dicom_record_uuid" in df.columns
    assert "metadata" in df.columns
    df.unpersist()
