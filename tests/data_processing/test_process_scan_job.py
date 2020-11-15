import pytest
from pytest_mock import mocker
import os, shutil, subprocess
from pyspark import SQLContext
from pyspark.sql.types import StringType,StructType,StructField
from click.testing import CliRunner
import os

from data_processing.common.sparksession import SparkConfig
from data_processing.common.Neo4jConnection import Neo4jConnection
from data_processing.process_scan_job import generate_scan_table, cli

current_dir = os.getcwd()

@pytest.fixture(autouse=True)
def spark(monkeypatch):
    print('------setup------')
    # setup env
    monkeypatch.setenv("MIND_ROOT_DIR", current_dir+"/tests/data_processing/testdata/data")
    monkeypatch.setenv("MIND_WORK_DIR", "/work/")
    monkeypatch.setenv("MIND_GPFS_DIR", current_dir+"/tests/data_processing/testdata")
    monkeypatch.setenv("GRAPH_URI", "bolt://localhost:7883")
    stream = os.popen('which python')
    pypath =stream.read().rstrip()
    monkeypatch.setenv("PYSPARK_PYTHON", pypath) # python in venv, need to update if running locally!
    monkeypatch.setenv("SPARK_MASTER_URL", "local[*]")
    monkeypatch.setenv("IO_SERVICE_HOST", "localhost")
    monkeypatch.setenv("IO_SERVICE_PORT", "5090")

    # start delta_io_service
    os.system("python3 -m data_processing.services.delta_io_service --hdfs file:/// --host localhost &")

    spark = SparkConfig().spark_session('tests/test_config.yaml', 'test-process-scan')
    yield spark

    print('------teardown------')
    work_dir = current_dir+"/tests/data_processing/testdata/work"
    if os.path.exists(work_dir):
        shutil.rmtree(work_dir)

    spark.stop()


def test_cli(mocker, spark):

    # mock neo4j
    mocker.patch('data_processing.common.Neo4jConnection.Neo4jConnection')
    mocker.patch.object(Neo4jConnection, 'commute_source_id_to_spark_query')

    sqlc = SQLContext(spark)
    cSchema = StructType([StructField('SeriesInstanceUID', StringType(), True)])
    ids = sqlc.createDataFrame([('1.2.840.113619.2.340.3.2743924432.468.1441191460.240',)],schema=cSchema)
    Neo4jConnection.commute_source_id_to_spark_query.return_value = ids

    runner = CliRunner()
    generate_mhd_script_path = os.path.join(current_dir, "data_processing/generateMHD.py")
    result = runner.invoke(cli, ['-q', "WHERE source:xnat_accession_number AND source.value='RIA_16-158_001' AND ALL(rel IN r WHERE TYPE(rel) IN ['ID_LINK'])",
        '-d', 'file:///',
        '-c', generate_mhd_script_path,
        '-t', 'scan.unittest',
        '-f', 'tests/test_config.yaml'])

    assert result.exit_code == 0
    # radiology.dcm in testdata has only 1 row
    Neo4jConnection.commute_source_id_to_spark_query.assert_called_once()
