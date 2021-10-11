import pytest
import os, shutil
from click.testing import CliRunner

from luna.common.Neo4jConnection import Neo4jConnection
from luna.services.graph_service import update_graph

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
        '-d', 'tests/luna/services/testdata/dicom-config.yaml',
        '-a', 'pyluna-core/tests/test_config.yml'])

    assert result.exit_code == 0


def test_cli_mha_table(mocker):

    # mock graph connection
    mocker.patch.object(Neo4jConnection, 'query')

    runner = CliRunner()
    result = runner.invoke(update_graph, [
        '-d', 'tests/luna/services/testdata/mha-config.yaml',
        '-a', 'pyluna-core/tests/test_config.yml'])

    assert result.exit_code == 0


def test_cli_mhd_table(mocker):

    # mock graph connection
    mocker.patch.object(Neo4jConnection, 'query')

    runner = CliRunner()
    result = runner.invoke(update_graph, [
        '-d', 'tests/luna/services/testdata/mhd-config.yaml',
        '-a', 'pyluna-core/tests/test_config.yml'])

    assert result.exit_code == 0


def test_cli_png_table(mocker):

    # mock graph connection
    mocker.patch.object(Neo4jConnection, 'query')

    runner = CliRunner()
    result = runner.invoke(update_graph, [
        '-d', 'tests/luna/services/testdata/png-config.yaml',
        '-a', 'pyluna-core/tests/test_config.yml'])

    assert result.exit_code == 0


def test_cli_feature_table(mocker):

    # mock graph connection
    mocker.patch.object(Neo4jConnection, 'query')

    runner = CliRunner()
    result = runner.invoke(update_graph, [
        '-d', 'tests/luna/services/testdata/feature-config.yaml',
        '-a', 'pyluna-core/tests/test_config.yml'])

    assert result.exit_code == 0

def test_cli_regional_bitmask_table(mocker):

    # mock graph connection
    mocker.patch.object(Neo4jConnection, 'query')

    runner = CliRunner()
    result = runner.invoke(update_graph, [
        '-d', 'tests/luna/services/testdata/regional_bitmask-config.yaml',
        '-a', 'pyluna-core/tests/test_config.yml'])

    assert result.exit_code == 0

def test_cli_regional_geojson_table(mocker):

    # mock graph connection
    mocker.patch.object(Neo4jConnection, 'query')

    runner = CliRunner()
    result = runner.invoke(update_graph, [
        '-d', 'tests/luna/services/testdata/regional_geojson-config.yaml',
        '-a', 'pyluna-core/tests/test_config.yml'])

    assert result.exit_code == 0

def test_cli_point_raw_json_table(mocker):

    # mock graph connection
    mocker.patch.object(Neo4jConnection, 'query')

    runner = CliRunner()
    result = runner.invoke(update_graph, [
        '-d', 'tests/luna/services/testdata/point_json-config.yaml',
        '-a', 'pyluna-core/tests/test_config.yml'])

    assert result.exit_code == 0

def test_cli_point_geojson_table(mocker):

    # mock graph connection
    mocker.patch.object(Neo4jConnection, 'query')

    runner = CliRunner()
    result = runner.invoke(update_graph, [
        '-d', 'tests/luna/services/testdata/point_geojson-config.yaml',
        '-a', 'pyluna-core/tests/test_config.yml'])

    assert result.exit_code == 0
