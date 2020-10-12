import pytest
from pytest_mock import mocker
from unittest import mock
import os, shutil
from pyspark import SQLContext
from pyspark.sql.types import StringType,StructType,StructField
from click.testing import CliRunner

from data_processing.common.sparksession import SparkConfig
from data_processing.common.Neo4jConnection import Neo4jConnection
from data_processing.process_scan_job import generate_scan_table, cli

current_dir = os.getcwd()

@pytest.fixture(autouse=True)
def spark():
    print('------setup------')
    spark = SparkConfig().spark_session('test-dicom-to-graph', 'local[2]')
    yield spark

    print('------teardown------')
    work_dir = current_dir+"/tests/data_processing/testdata/work"
    if os.path.exists(work_dir):
        shutil.rmtree(work_dir)


def test_cli(mocker, spark, monkeypatch):
    # setup env
    monkeypatch.setenv("MIND_ROOT_DIR", current_dir+"/tests/data_processing/testdata/data")
    monkeypatch.setenv("MIND_WORK_DIR", current_dir+"/tests/data_processing/testdata/work")
    monkeypatch.setenv("MIND_GPFS_DIR", current_dir+"/tests/data_processing/testdata")
    monkeypatch.setenv("GRAPH_URI", "bolt://localhost:7883")
    monkeypatch.setenv("PYSPARK_PYTHON", "/usr/bin/python3")
    monkeypatch.setenv("SPARK_MASTER_URL", "local[*]")
    monkeypatch.setenv("IO_SERVICE_HOST", "localhost")
    monkeypatch.setenv("IO_SERVICE_PORT", "5090")

    # mock neo4j
    mocker.patch('data_processing.common.Neo4jConnection.Neo4jConnection')

    sqlc = SQLContext(spark)
    mocker.patch.object(Neo4jConnection, 'commute_source_id_to_spark_query')
    cSchema = StructType([StructField('SeriesInstanceUID', StringType(), True)])
    ids = sqlc.createDataFrame([('1.2.840.113619.2.340.3.2743924432.468.1441191460.240',)],schema=cSchema)
    Neo4jConnection.commute_source_id_to_spark_query.return_value = ids


    runner = CliRunner()
    generate_mhd_script_path = os.path.join(current_dir, "data_processing/generateMHD.py")
    result = runner.invoke(cli, ['-q', "WHERE source:xnat_accession_number AND source.value='RIA_16-158_001' AND ALL(rel IN r WHERE TYPE(rel) IN ['ID_LINK'])",
        '-d', 'file:///',
        '-c', generate_mhd_script_path,
        '-t', 'scan.unittest'])

    print(result.exc_info)
    assert result.exit_code == 0
    # radiology.dcm in testdata has only 1 row
    Neo4jConnection.commute_source_id_to_spark_query.assert_called_once()