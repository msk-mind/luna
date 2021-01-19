import os, shutil
import pytest
from pytest_mock import mocker
from click.testing import CliRunner

from data_processing.scanManager.generateScan import cli as cli_generateScan

cwd = os.getcwd()

def test_cli_mhd(mocker, monkeypatch):
    monkeypatch.setenv("MIND_GPFS_DIR", cwd+"/tests/data_mock/")

    # mock graph connection
    mocker.patch("data_processing.scanManager.generateScan.get_container_data", return_value={'SeriesInstanceUID': "1.240.0.0", 'path':f'file:{cwd}/tests/data_processing/testdata/data/2.000000-CTAC-24716/dicoms/'})
    mocker.patch("data_processing.scanManager.generateScan.get_method_data", return_value={'file_ext':'mhd'})

    mock_db = mocker.patch("data_processing.scanManager.generateScan.add_container_data")

    runner = CliRunner()
    result = runner.invoke(cli_generateScan, [
        '-c', 'test-cohort',
        '-s', '1',
        '-m', 'test-method'])
    mock_db.assert_called_once()
#    assert "mhd-e0c2c6182c51052cffc1c4ae4f0e6c1af9a666998f3a291107c060344cf16f64" in mock_db.call_args_list[0][0][1].properties['QualifiedPath']
    assert result.exit_code == 0 

def test_cli_nrrd(mocker, monkeypatch):
    monkeypatch.setenv("MIND_GPFS_DIR", cwd+"/tests/data_mock/")

    # mock graph connection
    mocker.patch("data_processing.scanManager.generateScan.get_container_data", return_value={'SeriesInstanceUID': "1.240.0.0", 'path':f'file:{cwd}/tests/data_processing/testdata/data/2.000000-CTAC-24716/dicoms/'})
    mocker.patch("data_processing.scanManager.generateScan.get_method_data", return_value={'file_ext':'nrrd'})

    mock_db = mocker.patch("data_processing.scanManager.generateScan.add_container_data")

    runner = CliRunner()
    result = runner.invoke(cli_generateScan, [
        '-c', 'test-cohort',
        '-s', '1',
        '-m', 'test-method'])
    mock_db.assert_called_once()
#    assert "nrrd-e0c2c6182c51052cffc1c4ae4f0e6c1af9a666998f3a291107c060344cf16f64" in mock_db.call_args_list[0][0][1].properties['QualifiedPath']
    assert result.exit_code == 0 

