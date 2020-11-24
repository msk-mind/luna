import os
import shutil
import subprocess

import pytest
from click.testing import CliRunner
from data_processing.radiology.proxy_table.generate import *
from data_processing.common.sparksession import SparkConfig
import data_processing.common.constants as const

landing_path = "tests/data_processing/radiology/proxy_table/test_data/OV_16-158_CT_20201028/"
test_ingestion_template = "tests/data_processing/radiology/proxy_table/test_data/OV_16-158_CT_20201028/manifest.yaml"

@pytest.fixture(autouse=True)
def spark():
    print('------setup------')
    APP_CFG = 'APP_CFG'
    ConfigSet(name=APP_CFG, config_file='tests/test_config.yaml')
    spark = SparkConfig().spark_session(config_name=APP_CFG, app_name='test-radiology-proxy')

    yield spark

    print('------teardown------')
    tables_path = os.path.join(landing_path, "tables")
    if os.path.exists(tables_path):
        shutil.rmtree(tables_path)
    os.remove(test_ingestion_template)


def test_cli(spark):

    runner = CliRunner()
    result = runner.invoke(cli, 
        ['-t', 'tests/data_processing/data_ingestion_template_valid.yml',
        '-f', 'tests/test_config.yaml',
        '-p', 'delta'])

    print(result.exc_info)
    assert result.exit_code == 0

    df = spark.read.format("delta").load(landing_path + const.DICOM_TABLE)
    assert df.count() == 1
    assert "dicom_record_uuid" in df.columns
    assert "metadata" in df.columns
