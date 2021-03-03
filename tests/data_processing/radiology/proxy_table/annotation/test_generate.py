import os
import shutil
import subprocess

import pytest
from click.testing import CliRunner
from data_processing.radiology.proxy_table.annotation.generate import *
from data_processing.common.sparksession import SparkConfig
import data_processing.common.constants as const

mha_table_path = "tests/data_processing/radiology/testdata/test-project/tables/MHA_datasetname"
mhd_table_path = "tests/data_processing/radiology/testdata/test-project/tables/MHD_datasetname"

@pytest.fixture(autouse=True)
def spark():
    print('------setup------')
    ConfigSet(name=const.APP_CFG, config_file='tests/test_config.yaml')
    spark = SparkConfig().spark_session(config_name=const.APP_CFG, app_name='test-radiology-proxy-annotation')

    yield spark

    print('------teardown------')
    if os.path.exists(mha_table_path):
        shutil.rmtree(mha_table_path)
    if os.path.exists(mhd_table_path):
        shutil.rmtree(mhd_table_path)

def test_cli_mha(spark):

    runner = CliRunner()
    result = runner.invoke(cli, 
        ['-d', 'tests/data_processing/radiology/proxy_table/annotation/mha_data_config.yaml',
        '-a', 'tests/test_config.yaml',
        '-p', 'delta'])
    assert result.exit_code == 0

    df = spark.read.format("delta").load(os.path.join(os.getcwd(), mha_table_path))
    df.show(10, False)
    assert df.count() == 1
    columns = ['modificationTime', 'length','scan_annotation_record_uuid', 'path', 'accession_number', 'series_number', 'label', 'metadata']
    assert set(columns) \
            == set(df.columns)

    df.unpersist()

def test_cli_mhd(spark):

    runner = CliRunner()
    result = runner.invoke(cli, 
        ['-d', 'tests/data_processing/radiology/proxy_table/annotation/mhd_data_config.yaml',
        '-a', 'tests/test_config.yaml',
        '-p', 'delta'])
    assert result.exit_code == 0

    df = spark.read.format("delta").load(os.path.join(os.getcwd(), mhd_table_path))
    df.show(10, False)
    assert df.count() == 1
    columns = ['modificationTime', 'length','scan_annotation_record_uuid', 'path', 'accession_number', 'series_number', 'label', 'metadata']
    assert set(columns) \
            == set(df.columns)

    df.unpersist()
