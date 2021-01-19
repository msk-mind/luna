import os, shutil
import pytest
from pytest_mock import mocker
from click.testing import CliRunner

from data_processing.scanManager.extractRadiomics import cli as cli_generateScan

cwd = os.getcwd()
print (cwd)
def test_cli_radiomics(mocker, monkeypatch):
    monkeypatch.setenv("MIND_GPFS_DIR", cwd+"/tests/data_mock/")
    monkeypatch.setenv("GRAPH_URI", "bolt:/localhost:0000")

    # mock graph connection
    mocker.patch("data_processing.scanManager.extractRadiomics.get_container_data", return_value={'object.SeriesInstanceUID': "1.240.0.1", 'image.path':f'file:{cwd}/tests/data_processing/testdata/data/2.000000-CTAC-24716/volumes/image.mhd', 'label.path':f'file:{cwd}/tests/data_processing/testdata/data/2.000000-CTAC-24716/volumes/label.mha'})
    mocker.patch("data_processing.scanManager.extractRadiomics.get_method_data", return_value={'interpolator': 'sitkBSpline', 'resampledPixelSpacing': [1, 1, 1], 'padDistance': 10, 'voxelArrayShift': 1000, 'binWidth': 25, 'verbose': 'True', 'label': 1, 'geometryTolerance': 0.0001})

    mock_db = mocker.patch("data_processing.common.Neo4jConnection.Neo4jConnection.query", return_value='DB')

    runner = CliRunner()
    result = runner.invoke(cli_generateScan, [
        '-c', 'test-cohort',
        '-s', '1',
        '-m', 'test-method'])
    print (result.output)
    assert "RAD-7dc6e11804d9b9cea0e3e1ab296822f0d4a5372de7dd9eb3a98c860042d8a049" in mock_db.call_args_list[0][0][0]
    assert result.exit_code == 0 

def test_cli_radiomics_params(mocker, monkeypatch):
    monkeypatch.setenv("MIND_GPFS_DIR", cwd+"/tests/data_mock/")
    monkeypatch.setenv("GRAPH_URI", "bolt:/localhost:0000")

    # mock graph connection
    mocker.patch("data_processing.scanManager.extractRadiomics.get_container_data", return_value={'object.SeriesInstanceUID': "1.240.0.1", 'image.path':f'file:{cwd}/tests/data_processing/testdata/data/2.000000-CTAC-24716/volumes/image.mhd', 'label.path':f'file:{cwd}/tests/data_processing/testdata/data/2.000000-CTAC-24716/volumes/label.mha'})
    mocker.patch("data_processing.scanManager.extractRadiomics.get_method_data", return_value={'interpolator': 'sitkBSpline', 'resampledPixelSpacing': [1, 1, 1], 'padDistance': 10, 'voxelArrayShift': 1000, 'binWidth': 50, 'verbose': 'True', 'label': 1, 'geometryTolerance': 0.0001})

    mock_db = mocker.patch("data_processing.common.Neo4jConnection.Neo4jConnection.query", return_value='DB')

    runner = CliRunner()
    result = runner.invoke(cli_generateScan, [
        '-c', 'test-cohort',
        '-s', '1',
        '-m', 'test-method'])
    print (result.output)
    assert "RAD-00c4f67b5618b12f02466c6dc75e969718b2e174e137f6c992606a17f2737725" in mock_db.call_args_list[0][0][0]
    assert result.exit_code == 0 

