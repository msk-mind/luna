import os
import shutil
import subprocess

import pytest
from click.testing import CliRunner
from data_processing.radiology.proxy_table.annotation.generate import *
from data_processing.common.sparksession import SparkConfig
import data_processing.common.constants as const

table_path = "tests/data_processing/testdata/data/test-project/tables/MHA_datasetname"

@pytest.fixture(autouse=True)
def spark():
    print('------setup------')
    ConfigSet(name=const.APP_CFG, config_file='tests/test_config.yaml')
    spark = SparkConfig().spark_session(config_name=const.APP_CFG, app_name='test-radiology-proxy-annotation')

    yield spark

    print('------teardown------')
    if os.path.exists(table_path):
        shutil.rmtree(table_path)


def test_cli(spark):

    runner = CliRunner()
    result = runner.invoke(cli, 
        ['-t', 'tests/data_processing/radiology/proxy_table/test_data/annotations.yaml',
        '-f', 'tests/test_config.yaml',
        '-p', 'delta'])
    assert result.exit_code == 0

    df = spark.read.format("delta").load(os.path.join(os.getcwd(), table_path))
    df.show(10, False)
    assert df.count() == 1
    assert set(['modificationTime', 'length','scan_annotation_record_uuid', 'path', 'accession_number', 'metadata']) \
            == set(df.columns)

    df.unpersist()
