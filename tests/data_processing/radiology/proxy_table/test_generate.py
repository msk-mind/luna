import os
import shutil
import subprocess

import pytest
from click.testing import CliRunner
from data_processing.radiology.proxy_table.generate import *
from data_processing.common.sparksession import SparkConfig


dicom_table_path = "tests/data_processing/radiology/proxy_table/test_data/OV_16-158_CT_20201028/table/"
test_ingestion_template = "tests/data_processing/radiology/proxy_table/test_data/OV_16-158_CT_20201028/manifest.yaml"

@pytest.fixture(autouse=True)
def spark():
    print('------setup------')
    spark = SparkConfig().spark_session('tests/data_processing/common/test_config.yaml',
                                        'test-radiology-proxy')
    yield spark

    print('------teardown------')
    if os.path.exists(dicom_table_path):
        shutil.rmtree(dicom_table_path)
    os.remove(test_ingestion_template)


def test_cli(spark):

    runner = CliRunner()
    result = runner.invoke(cli, 
        ['-t', 'tests/data_processing/data_ingestion_template_valid.yaml', 
        '-f', 'tests/data_processing/common/test_config.yaml', 
        '-p', 'delta'])

    print(result.exc_info)
    assert result.exit_code == 0

    df = spark.read.format("delta").load(dicom_table_path + "dicom")
    assert df.count() == 1
    assert "dicom_record_uuid" in df.columns
    assert "metadata" in df.columns
