import pytest
from click.testing import CliRunner

from data_processing.pathology.cli.dsa.dsa_viz import cli
from data_processing.pathology.cli.dsa.dsa_upload import cli as upload


def test_stardist_polygon(monkeypatch):

    runner = CliRunner()
    result = runner.invoke(cli,
                           ["-s", "stardist-polygon",
                            "-d","tests/data_processing/pathology/cli/dsa/testdata/stardist_polygon.json"])

    assert result.exit_code == 0


def test_stardist_cell(monkeypatch):

    runner = CliRunner()
    result = runner.invoke(cli,
                           ["-s", "stardist-cell",
                            "-d","tests/data_processing/pathology/cli/dsa/testdata/stardist_cell.json"])

    assert result.exit_code == 0


def test_regional_polygon(monkeypatch):

    runner = CliRunner()
    result = runner.invoke(cli,
                           ["-s", "regional-polygon",
                            "-d","tests/data_processing/pathology/cli/dsa/testdata/regional_polygon.json"])

    assert result.exit_code == 0


def test_heatmap():

    runner = CliRunner()
    result = runner.invoke(cli,
                           ["-s", "heatmap",
                            "-d","tests/data_processing/pathology/cli/dsa/testdata/heatmap_config.json"])

    assert result.exit_code == 0

def test_upload(requests_mock):

    requests_mock.get('http://localhost:8080/api/v1/system/check?mode=basic', text='{}')
    requests_mock.get('http://localhost:8080/api/v1/collection?text=test_collection&limit=5&sort=name&sortdir=1', text='[{\"name\": \"test_collection\", \"_id\": \"uuid\", \"_modelType\":\"collection\"}]')
    requests_mock.get('http://localhost:8080/api/v1/item?text=123&limit=50&sort=lowerName&sortdir=1', text='[{\"annotation\": {\"name\": \"123.svs\"}, \"_id\": \"uuid\", \"_modelType\":\"annotation\"}]')
    requests_mock.post('http://localhost:8080/api/v1/annotation?itemId=None', text='{}')

    runner = CliRunner()
    result = runner.invoke(upload,
                           ["-c", "tests/data_processing/pathology/cli/dsa/testdata/dsa_config.json",
                            "-d","tests/data_processing/pathology/cli/dsa/testdata/bitmask_polygon_upload.json"])

    assert result.exit_code == 0
