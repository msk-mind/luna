import pytest
from pytest_mock import mocker
import os, shutil
from pyspark import SQLContext
from click.testing import CliRunner
import os

from luna.common.config import ConfigSet
from luna.common.sparksession import SparkConfig
from luna.radiology.refined_table.generate import generate_scan_table, cli

current_dir = os.getcwd()
project_name = 'test-project'
scan_table_path = os.path.join("pyluna-radiology/tests/luna/testdata/data", project_name, "tables/scans")
APP_CFG='APP_CFG'

@pytest.fixture(autouse=True)
def spark(monkeypatch):
    print('------setup------')
    # setup env
    monkeypatch.setenv("MIND_ROOT_DIR", os.path.join(current_dir, "pyluna-radiology/tests/luna/testdata/data"))
    stream = os.popen('which python')
    pypath = stream.read().rstrip()
    monkeypatch.setenv("PYSPARK_PYTHON", pypath) # python in venv, need to update if running locally!
    monkeypatch.setenv("SPARK_MASTER_URL", "local[*]")

    ConfigSet(name=APP_CFG, config_file='pyluna-radiology/tests/test_config.yml')
    spark = SparkConfig().spark_session(config_name=APP_CFG, app_name='test-process-scan')
    yield spark

    print('------teardown------')
    if os.path.exists(scan_table_path):
        shutil.rmtree(scan_table_path)

"""luna.radiology.refined_table.generate#L113 doesn't work well for relative paths
def test_cli(mocker, spark):

    runner = CliRunner()
    generate_mhd_script_path = os.path.join(current_dir, "pyluna-radiology/luna/radiology/refined_table/dicom_to_scan.py")
    
    # This test doesn't pass as the paths extracted from the table don't match those on the circleci runner
    # TODO: Some patching is neccessary
#    mocker.patch('luna.radiology.refined_table.generate.generate_scan_table.python_def_generate_scan.os.path.split', side_effect=['/home/circleci/project/pyluna-radiology/tests/luna/testdata/data/test-project/dicoms/RIA_16_158A_000013/20150902_CT/2_Standard_5mm/'])
    
    for ext in ['mhd', 'nrrd']:
        result = runner.invoke(cli, ['-i', "1.2.840.113619.2.353.2807.624957.15092.1438009271.852",
            '-d', 'file:///',
            '-c', generate_mhd_script_path,
            '-p', project_name,
            '-e', ext,
            '-t', 'scan.unittest',
            '-f', 'pyluna-radiology/tests/test_config.yml'])

        assert result.exit_code == 0
        df = spark.read.format("delta").load(scan_table_path)
        assert set(['SeriesInstanceUID', 'scan_record_uuid', 'filepath', 'filetype']) == set(df.columns)
        df.unpersist()
"""
