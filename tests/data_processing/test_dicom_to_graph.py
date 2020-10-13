import pytest
from pytest_mock import mocker
from unittest import mock
import os
from click.testing import CliRunner

from data_processing.common.sparksession import SparkConfig
from data_processing.common.Neo4jConnection import Neo4jConnection
from data_processing.dicom_to_graph import update_graph_with_scans, cli

os.environ["MIND_ROOT_DIR"] = os.getcwd() + "/tests/data_processing/testdata/data"
os.environ["MIND_WORK_DIR"] = os.getcwd() + "/tests/data_processing/testdata/work"

@pytest.fixture(autouse=True)
def spark():
    print('------setup------')
    spark = SparkConfig().spark_session('tests/data_processing/common/test_config.yaml', 'test-dicom-to-graph')
    yield spark

    print('------teardown------')


def test_cli(mocker, spark):
    # mock neo4j
    mocker.patch('data_processing.common.Neo4jConnection.Neo4jConnection')
    Neo4jConnection.__init__.return_value = mock.Mock()
    mocker.patch.object(Neo4jConnection, 'query')
    Neo4jConnection.query.return_value = []

    runner = CliRunner()
    result = runner.invoke(cli, ['-s', 'local[2]', '-g', 'bolt://localhost:7883', '-h', 'file:///'])

    assert result.exit_code == 0
    # radiology.dcm in testdata has only 1 row
    Neo4jConnection.query.assert_called_once()
