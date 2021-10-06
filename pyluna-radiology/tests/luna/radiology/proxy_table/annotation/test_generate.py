import os
import shutil

import pytest
from click.testing import CliRunner
from luna.radiology.proxy_table.annotation.generate import *
from luna.common.sparksession import SparkConfig
import luna.common.constants as const
from luna.common.config import ConfigSet

project_path = "pyluna-radiology/tests/luna/radiology/testdata/test-project"
mha_table_path = project_path + "/tables/MHA_datasetname"
mha_app_config_path = project_path + "/configs/MHA_datasetname/app_config.yaml"
mha_data_config_path = project_path + "/configs/MHA_datasetname/data_config.yaml"
mha_data_csv_path = project_path + "/configs/MHA_datasetname/metadata.csv"

mhd_table_path = project_path + "/tables/RADIOLOGY_ANNOTATION_datasetname"
mhd_app_config_path = project_path + "/configs/RADIOLOGY_ANNOTATION_datasetname/app_config.yaml"
mhd_data_config_path = project_path + "/configs/RADIOLOGY_ANNOTATION_datasetname/data_config.yaml"
mhd_data_csv_path = project_path + "/configs/RADIOLOGY_ANNOTATION_datasetname/metadata.csv"

@pytest.fixture(autouse=True)
def spark():
    print('------setup------')
    ConfigSet(name=const.APP_CFG, config_file='pyluna-radiology/tests/test_config.yml')
    spark = SparkConfig().spark_session(config_name=const.APP_CFG, app_name='test-radiology-proxy-annotation')

    yield spark

    print('------teardown------')
    if os.path.exists(mha_table_path):
        shutil.rmtree(mha_table_path)
    if os.path.exists(mhd_table_path):
        shutil.rmtree(mhd_table_path)
    if os.path.exists(project_path+"/configs"):
        shutil.rmtree(project_path+"/configs")

def test_cli_mha(spark):

    runner = CliRunner()
    result = runner.invoke(cli, 
        ['-d', 'pyluna-radiology/tests/luna/radiology/proxy_table/annotation/mha_data_config.yaml',
        '-a', 'pyluna-radiology/tests/test_config.yml',
        '-p', 'delta'])
    print(result.exc_info)
    assert result.exit_code == 0

    assert os.path.exists(mha_app_config_path)
    assert os.path.exists(mha_data_config_path)
    assert os.path.exists(mha_data_csv_path)

    df = spark.read.format("delta").load(os.path.join(os.getcwd(), mha_table_path))
    df.show(10, False)
    assert df.count() == 1
    columns = ['modificationTime', 'length','scan_annotation_record_uuid', 'path', 'accession_number', 'series_number', 'label', 'metadata']
    assert set(columns) \
            == set(df.columns)

    df.unpersist()

def test_cli_both(spark):

    runner = CliRunner()
    result = runner.invoke(cli, 
        ['-d', 'pyluna-radiology/tests/luna/radiology/proxy_table/annotation/data_config.yaml',
        '-a', 'pyluna-radiology/tests/test_config.yml',
        '-p', 'delta'])
    print(result.exc_info)
    assert result.exit_code == 0

    assert os.path.exists(mhd_app_config_path)
    assert os.path.exists(mhd_data_config_path)
    assert os.path.exists(mhd_data_csv_path)

    df = spark.read.format("delta").load(os.path.join(os.getcwd(), mhd_table_path))
    df.show(10, False)
    assert df.count() == 2
    columns = ['modificationTime', 'length','scan_annotation_record_uuid', 'path', 'accession_number', 'series_number', 'label', 'metadata']
    assert set(columns) \
            == set(df.columns)

    df.unpersist()
