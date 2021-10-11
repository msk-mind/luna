import pytest
from pytest_mock import mocker
import os
from click.testing import CliRunner

from luna.common.config import ConfigSet
from luna.common.sparksession import SparkConfig
from luna.common.Neo4jConnection import Neo4jConnection
from luna.dicom_to_graph import update_graph_with_scans, cli

current_dir = os.getcwd()

@pytest.fixture(autouse=True)
def spark():
    print('------setup------')
    APP_CFG = 'APP_CFG'
    ConfigSet(name=APP_CFG, config_file='pyluna-core/tests/test_config.yml')
    spark = SparkConfig().spark_session(config_name=APP_CFG, app_name='test-dicom-to-graph')

    yield spark

    print('------teardown------')


def test_cli(mocker, spark, monkeypatch):
    # setup env
    monkeypatch.setenv("MIND_ROOT_DIR", current_dir+"/tests/luna/testdata/data")
    monkeypatch.setenv("MIND_WORK_DIR", current_dir+"/tests/luna/testdata/work")

    # mock neo4j
    mocker.patch('luna.common.Neo4jConnection.Neo4jConnection')
    mocker.patch.object(Neo4jConnection, 'query')
    Neo4jConnection.query.return_value = []

    runner = CliRunner()
    result = runner.invoke(cli, ['-s', 'local[2]',
                                 '-g', 'bolt://localhost:7883',
                                 '-h', 'file:///',
                                 '-f', 'pyluna-core/tests/test_config.yml'])

    assert result.exit_code == 0
    # radiology.dcm in testdata has only 1 row
    Neo4jConnection.query.assert_called_once()
