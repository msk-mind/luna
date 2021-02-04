import pytest
import os, shutil
from click.testing import CliRunner

from data_processing.common.Neo4jConnection import Neo4jConnection
from data_processing.services.graph_service import update_graph

@pytest.fixture(autouse=True)
def spark(monkeypatch):
    print('------setup------')
    # setup env
    stream = os.popen('which python')
    pypath = stream.read().rstrip()
    monkeypatch.setenv("PYSPARK_PYTHON", pypath)

    yield

    print('------teardown------')


def test_cli_dicom_table(mocker):

    # mock graph connection
    mocker.patch.object(Neo4jConnection, 'query')

    runner = CliRunner()
    result = runner.invoke(update_graph, [
        '-d', 'tests/data_processing/services/testdata/dicom-config.yaml',
        '-f', 'tests/test_config.yaml'])

    assert result.exit_code == 0


def test_cli_mha_table(mocker):

    # mock graph connection
    mocker.patch.object(Neo4jConnection, 'query')

    runner = CliRunner()
    result = runner.invoke(update_graph, [
        '-d', 'tests/data_processing/services/testdata/mha-config.yaml',
        '-f', 'tests/test_config.yaml'])

    assert result.exit_code == 0


def test_cli_mhd_table(mocker):

    # mock graph connection
    mocker.patch.object(Neo4jConnection, 'query')

    runner = CliRunner()
    result = runner.invoke(update_graph, [
        '-d', 'tests/data_processing/services/testdata/mhd-config.yaml',
        '-f', 'tests/test_config.yaml'])

    assert result.exit_code == 0


def test_cli_png_table(mocker):

    # mock graph connection
    mocker.patch.object(Neo4jConnection, 'query')

    runner = CliRunner()
    result = runner.invoke(update_graph, [
        '-d', 'tests/data_processing/services/testdata/png-config.yaml',
        '-f', 'tests/test_config.yaml'])

    assert result.exit_code == 0


def test_cli_feature_table(mocker):

    # mock graph connection
    mocker.patch.object(Neo4jConnection, 'query')

    runner = CliRunner()
    result = runner.invoke(update_graph, [
        '-d', 'tests/data_processing/services/testdata/feature-config.yaml',
        '-f', 'tests/test_config.yaml'])

    assert result.exit_code == 0

def test_cli_regional_bitmask_table(mocker):

    # mock graph connection
    mocker.patch.object(Neo4jConnection, 'query')

    runner = CliRunner()
    result = runner.invoke(update_graph, [
        '-d', 'tests/data_processing/services/testdata/regional_bitmask-config.yaml',
        '-f', 'tests/test_config.yaml'])

    assert result.exit_code == 0

def test_cli_regional_geojson_table(mocker):

    # mock graph connection
    mocker.patch.object(Neo4jConnection, 'query')

    runner = CliRunner()
    result = runner.invoke(update_graph, [
        '-d', 'tests/data_processing/services/testdata/regional_geojson-config.yaml',
        '-f', 'tests/test_config.yaml'])

    assert result.exit_code == 0
