import pytest
from pytest_mock import mocker
import os, shutil
from click.testing import CliRunner
from neo4j import Record

from data_processing.common.config import ConfigSet
from data_processing.common.Neo4jConnection import Neo4jConnection
from data_processing.common import constants as const
from data_processing.common.sparksession import SparkConfig
from data_processing.services.graph_service import update_graph

current_dir = os.getcwd()
project_name = 'test-project'

@pytest.fixture(autouse=True)
def spark(monkeypatch):
    print('------setup------')
    # setup env
    #monkeypatch.setenv("MIND_ROOT_DIR", os.path.join(current_dir, "tests/data_processing/testdata/data"))
    stream = os.popen('which python')
    pypath = stream.read().rstrip()
    monkeypatch.setenv("PYSPARK_PYTHON", pypath)

    ConfigSet(name=const.APP_CFG, config_file='tests/test_config.yaml')
    spark = SparkConfig().spark_session(config_name=const.APP_CFG, app_name='test-graph-service')
    yield spark

    print('------teardown------')


def test_cli_dicom_table(mocker, spark):

    # mock graph connection
    mocker.patch.object(Neo4jConnection, 'query')

    runner = CliRunner()
    result = runner.invoke(update_graph, [
        '-p', project_name,
        '-t', 'dicom',
        '-d', 'tests/data_processing/services/config.yaml',
        '-f', 'tests/test_config.yaml'])

    assert result.exit_code == 0

