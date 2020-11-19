import pytest
from pytest_mock import mocker
import os, shutil
from pyspark import SQLContext
from pyspark.sql.types import StringType,StructType,StructField
from click.testing import CliRunner
import os

from data_processing.common.sparksession import SparkConfig
from data_processing.common.Neo4jConnection import Neo4jConnection
from data_processing.radiology.refined_table.generate import generate_scan_table, cli

current_dir = os.getcwd()
project_name = 'test-project'
scan_table_path = os.path.join(current_dir, "tests/data_processing/testdata/data", project_name, "tables/scan")

@pytest.fixture(autouse=True)
def spark(monkeypatch):
    print('------setup------')
    # setup env
    monkeypatch.setenv("MIND_ROOT_DIR", os.path.join(current_dir, "tests/data_processing/testdata/data"))
    monkeypatch.setenv("MIND_GPFS_DIR", os.path.join(current_dir, "tests/data_processing/testdata"))
    monkeypatch.setenv("GRAPH_URI", "bolt://localhost:7883")
    stream = os.popen('which python')
    pypath = stream.read().rstrip()
    monkeypatch.setenv("PYSPARK_PYTHON", pypath) # python in venv, need to update if running locally!
    monkeypatch.setenv("SPARK_MASTER_URL", "local[*]")

    spark = SparkConfig().spark_session('tests/test_config.yaml', 'test-process-scan')
    yield spark

    print('------teardown------')
    if os.path.exists(scan_table_path):
        shutil.rmtree(scan_table_path)
    spark.stop()


def test_cli(mocker, spark):

    # mock neo4j
    """mocker.patch('data_processing.common.Neo4jConnection.Neo4jConnection')
    mocker.patch.object(Neo4jConnection, 'commute_source_id_to_spark_query')

    sqlc = SQLContext(spark)
    cSchema = StructType([StructField('SeriesInstanceUID', StringType(), True)])
    ids = sqlc.createDataFrame([('1.2.840.113619.2.340.3.2743924432.468.1441191460.240',)],schema=cSchema)
    Neo4jConnection.commute_source_id_to_spark_query.return_value = ids"""

    runner = CliRunner()
    generate_mhd_script_path = os.path.join(current_dir, "data_processing/radiology/refined_table/dicom_to_scan.py")

    for ext in ['mhd', 'nrrd']:
        result = runner.invoke(cli, ['-i', "1.2.840.113619.2.340.3.2743924432.468.1441191460.240",
            '-d', 'file:///',
            '-c', generate_mhd_script_path,
            '-p', project_name,
            '-e', ext,
            '-t', 'scan.unittest',
            '-f', 'tests/test_config.yaml'])

        assert result.exit_code == 0
        df = spark.read.format("delta").load('file:///'+scan_table_path)
        assert set(['SeriesInstanceUID', 'scan_record_uuid', 'filepath', 'filetype']) == set(df.columns)
    # radiology.dcm in testdata has only 1 row
    #Neo4jConnection.commute_source_id_to_spark_query.assert_called_once()
