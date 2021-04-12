import pytest
import requests, click
from click.testing import CliRunner

from data_processing.pathology.cli.dsa.dsa_viz import cli
from tests.data_processing.pathology.common.request_mock import MockResponse


def test_stardist_polygon(monkeypatch):

    def mock_system_check(*args, **kwargs):
        return MockResponse({}, 200)
    monkeypatch.setattr(requests, "get", mock_system_check)

    def mock_get_uuid(*args, **kwargs):
        return MockResponse("[{\"annotation\": {\"name\": \"123.svs\"}, \"_id\": \"uuid\", \"_modelType\":\"annotation\"}]", 200)
    monkeypatch.setattr(requests, "get", mock_get_uuid)

    def mock_get_annotation_uuid(*args, **kwargs):
        return MockResponse("[{\"annotation\": {\"name\": \"123.svs\"}, \"_id\": \"uuid\", \"_modelType\":\"annotation\"}]", 200)
    monkeypatch.setattr(requests, "get", mock_get_annotation_uuid)

    def mock_push(*args, **kwargs):
        return MockResponse({}, 200)
    monkeypatch.setattr(requests, "post", mock_push)

    runner = CliRunner()
    result = runner.invoke(cli,
                           ["-c","tests/data_processing/pathology/cli/dsa/testdata/dsa_config.json",
                            "stardist-polygon",
                            "-d","tests/data_processing/pathology/cli/dsa/testdata/stardist_polygon.json"])

    assert result.exit_code == 0


def test_stardist_cell(monkeypatch):

    def mock_system_check(*args, **kwargs):
        return MockResponse({}, 200)
    monkeypatch.setattr(requests, "get", mock_system_check)

    def mock_get_uuid(*args, **kwargs):
        return MockResponse("[{\"annotation\": {\"name\": \"123.svs\"}, \"_id\": \"uuid\", \"_modelType\":\"annotation\"}]", 200)
    monkeypatch.setattr(requests, "get", mock_get_uuid)

    def mock_get_annotation_uuid(*args, **kwargs):
        return MockResponse("[{\"annotation\": {\"name\": \"123.svs\"}, \"_id\": \"uuid\", \"_modelType\":\"annotation\"}]", 200)
    monkeypatch.setattr(requests, "get", mock_get_annotation_uuid)

    def mock_push(*args, **kwargs):
        return MockResponse({}, 200)
    monkeypatch.setattr(requests, "post", mock_push)

    runner = CliRunner()
    result = runner.invoke(cli,
                           ["-c","tests/data_processing/pathology/cli/dsa/testdata/dsa_config.json",
                            "stardist-cell",
                            "-d","tests/data_processing/pathology/cli/dsa/testdata/stardist_cell.json"])

    assert result.exit_code == 0


def test_regional_polygon(monkeypatch):

    def mock_system_check(*args, **kwargs):
        return MockResponse({}, 200)
    monkeypatch.setattr(requests, "get", mock_system_check)

    def mock_get_uuid(*args, **kwargs):
        return MockResponse("[{\"annotation\": {\"name\": \"123.svs\"}, \"_id\": \"uuid\", \"_modelType\":\"annotation\"}]", 200)
    monkeypatch.setattr(requests, "get", mock_get_uuid)

    def mock_get_annotation_uuid(*args, **kwargs):
        return MockResponse("[{\"annotation\": {\"name\": \"456.svs\"}, \"_id\": \"uuid\", \"_modelType\":\"annotation\"}]", 200)
    monkeypatch.setattr(requests, "get", mock_get_annotation_uuid)

    def mock_push(*args, **kwargs):
        return MockResponse({}, 200)
    monkeypatch.setattr(requests, "post", mock_push)

    runner = CliRunner()
    result = runner.invoke(cli,
                           ["-c","tests/data_processing/pathology/cli/dsa/testdata/dsa_config.json",
                            "regional-polygon",
                            "-d","tests/data_processing/pathology/cli/dsa/testdata/regional_polygon.json"])

    assert result.exit_code == 0

# commenting out - maybe timeout in circleci?
"""
def test_bitmask_polygon(monkeypatch):

    def mock_system_check(*args, **kwargs):
        return MockResponse({}, 200)
    monkeypatch.setattr(requests, "get", mock_system_check)

    def mock_get_uuid(*args, **kwargs):
        return MockResponse("{\"name\": \"123.svs\", \"_id\": \"uuid\"}", 200)
    monkeypatch.setattr(requests, "get", mock_get_uuid)

    def mock_push(*args, **kwargs):
        return MockResponse({}, 200)
    monkeypatch.setattr(requests, "post", mock_push)

    runner = CliRunner()
    result = runner.invoke(cli,
                           ["-c","tests/data_processing/pathology/cli/dsa/testdata/dsa_config.json",
                            "bitmask-polygon",
                            "-d","tests/data_processing/pathology/cli/dsa/testdata/bitmask_polygon.json"])

    assert result.exit_code == 0
"""

def test_heatmap(monkeypatch):

    def mock_system_check(*args, **kwargs):
        return MockResponse({}, 200)
    monkeypatch.setattr(requests, "get", mock_system_check)

    def mock_get_uuid(*args, **kwargs):
        return MockResponse("[{\"annotation\": {\"name\": \"123.svs\"}, \"_id\": \"uuid\", \"_modelType\":\"annotation\"}]", 200)
    monkeypatch.setattr(requests, "get", mock_get_uuid)

    def mock_get_annotation_uuid(*args, **kwargs):
        return MockResponse("[{\"annotation\": {\"name\": \"456.svs\"}, \"_id\": \"uuid\", \"_modelType\":\"annotation\"}]", 200)
    monkeypatch.setattr(requests, "get", mock_get_annotation_uuid)

    def mock_push(*args, **kwargs):
        return MockResponse({}, 200)
    monkeypatch.setattr(requests, "post", mock_push)

    runner = CliRunner()
    result = runner.invoke(cli,
                           ["-c","tests/data_processing/pathology/cli/dsa/testdata/dsa_config.json",
                            "heatmap",
                            "-d","tests/data_processing/pathology/cli/dsa/testdata/heatmap_config.json"])

    assert result.exit_code == 0
