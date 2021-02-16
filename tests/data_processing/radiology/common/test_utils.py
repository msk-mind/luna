import pytest
from data_processing.radiology.common.utils import *
from data_processing.common.sparksession import SparkConfig
from data_processing.common.config import ConfigSet
import data_processing.common.constants as const

dicom_path = "tests/data_processing/testdata/data/test-project/dicoms/1.dcm"
mha_path = "tests/data_processing/testdata/data/test-project/scan_annotations/346677/tumor-label.mha"
mhd_path = "tests/data_processing/testdata/data/test-project/scan_annotations/346677/volumetric_seg.mhd"

def test_find_centroid():

    xy = find_centroid(mha_path, 512, 512)
    assert 360 == xy[0]
    assert 198 == xy[1]

def test_dicom_to_bytes():

    image = dicom_to_bytes(dicom_path, 512, 512)

    assert bytes == type(image)
    assert 512*512 == len(image)

def test_create_seg_images():

    arr = create_seg_images(mhd_path, "uuid", 512, 512)

    assert 21 == len(arr)
    assert "uuid" == arr[0][1]
    # RGB seg image
    assert 512*512*3 == len(arr[0][2])


def test_crop_images():

    ConfigSet(name=const.APP_CFG, config_file='tests/test_config.yaml')
    spark = SparkConfig().spark_session(config_name=const.APP_CFG, app_name='test-radiology-utils')

    png = spark.read.format("delta").load("tests/data_processing/testdata/data/test-project/tables/PNG_dsn").toPandas()
    dicom = png['dicom'][0]
    overlay = png['overlay'][0]
    dicom_overlay = crop_images(360, 198, dicom, overlay, 256, 256, 512, 512)

    assert 256*256 == len(dicom_overlay[0])
    # RGB overlay image
    assert 256*256*3 == len(dicom_overlay[1])










# def test_cli_radiomics(mocker, monkeypatch):
#     monkeypatch.setenv("MIND_GPFS_DIR", cwd+"/tests/data_mock/")

#     # mock graph connection
#     #mocker.patch("data_processing.scanManager.extractRadiomics.get_container_data", return_value={'container.QualifiedPath':'test-cohort::1.240.0.1', 'container.name': "1.240.0.1", 'image.path':f'file:{cwd}/tests/data_processing/testdata/data/2.000000-CTAC-24716/volumes/image.mhd', 'label.path':f'file:{cwd}/tests/data_processing/testdata/data/2.000000-CTAC-24716/volumes/label.mha'})
#     mocker.patch("data_processing.common.utils.get_method_data", return_value={'interpolator': 'sitkBSpline', 'resampledPixelSpacing': [1, 1, 1], 'padDistance': 10, 'voxelArrayShift': 1000, 'binWidth': 25, 'verbose': 'True', 'label': 1, 'geometryTolerance': 0.0001})

#     #mock_db = mocker.patch("data_processing.scanManager.extractRadiomics.add_container_data", return_value=None)

#     runner = CliRunner()
#     result = runner.invoke(cli_extractRadiomics, [
#         '-c', 'test-cohort',
#         '-s', '1',
#         '-m', 'test-method'])
#     mock_db.assert_called_once()
# #    assert "RAD-7dc6e11804d9b9cea0e3e1ab296822f0d4a5372de7dd9eb3a98c860042d8a049" in mock_db.call_args_list[0][0][1].properties['QualifiedPath']
#     assert result.exit_code == 0 

# def test_cli_radiomics_params(mocker, monkeypatch):
#     monkeypatch.setenv("MIND_GPFS_DIR", cwd+"/tests/data_mock/")

#     # mock graph connection
#     #mocker.patch("data_processing.scanManager.extractRadiomics.get_container_data", return_value={'container.QualifiedPath':'test-cohort::1.240.0.2','container.name': "1.240.0.2", 'image.path':f'file:{cwd}/tests/data_processing/testdata/data/2.000000-CTAC-24716/volumes/image.mhd', 'label.path':f'file:{cwd}/tests/data_processing/testdata/data/2.000000-CTAC-24716/volumes/label.mha'})
#     mocker.patch("data_processing.common.utils.get_method_data", return_value={'interpolator': 'sitkBSpline', 'resampledPixelSpacing': [1, 1, 1], 'padDistance': 10, 'voxelArrayShift': 1000, 'binWidth': 50, 'verbose': 'True', 'label': 1, 'geometryTolerance': 0.0001})

#     #mock_db = mocker.patch("data_processing.scanManager.extractRadiomics.add_container_data", return_value=None)

#     runner = CliRunner()
#     result = runner.invoke(cli_extractRadiomics, [
#         '-c', 'test-cohort',
#         '-s', '1',
#         '-m', 'test-method'])
#     mock_db.assert_called_once()
# #    assert "RAD-00c4f67b5618b12f02466c6dc75e969718b2e174e137f6c992606a17f2737725" in mock_db.call_args_list[0][0][1].properties['QualifiedPath']
#     assert result.exit_code == 0 



# import os, shutil
# import pytest
# from pytest_mock import mocker
# from click.testing import CliRunner

# from data_processing.scanManager.generateScan import cli as cli_generateScan

# cwd = os.getcwd()

# def test_cli_mhd(mocker, monkeypatch):
#     monkeypatch.setenv("MIND_GPFS_DIR", cwd+"/tests/data_mock/")

#     # mock graph connection
#     mocker.patch("data_processing.scanManager.generateScan.get_container_data", return_value={'SeriesInstanceUID': "1.240.0.0", 'path':f'file:{cwd}/tests/data_processing/testdata/data/2.000000-CTAC-24716/dicoms/'})
#     mocker.patch("data_processing.scanManager.generateScan.get_method_data", return_value={'file_ext':'mhd'})

#     mock_db = mocker.patch("data_processing.scanManager.generateScan.add_container_data", return_value=None)

#     runner = CliRunner()
#     result = runner.invoke(cli_generateScan, [
#         '-c', 'test-cohort',
#         '-s', '1',
#         '-m', 'test-method'])
#     mock_db.assert_called_once()
#     assert "mhd-e0c2c6182c51052cffc1c4ae4f0e6c1af9a666998f3a291107c060344cf16f64" in mock_db.call_args_list[0][0][1].properties['QualifiedPath']
#     assert result.exit_code == 0 

# def test_cli_nrrd(mocker, monkeypatch):
#     monkeypatch.setenv("MIND_GPFS_DIR", cwd+"/tests/data_mock/")

#     # mock graph connection
#     mocker.patch("data_processing.scanManager.generateScan.get_container_data", return_value={'SeriesInstanceUID': "1.240.0.0", 'path':f'file:{cwd}/tests/data_processing/testdata/data/2.000000-CTAC-24716/dicoms/'})
#     mocker.patch("data_processing.scanManager.generateScan.get_method_data", return_value={'file_ext':'nrrd'})

#     mock_db = mocker.patch("data_processing.scanManager.generateScan.add_container_data", return_value=None)

#     runner = CliRunner()
#     result = runner.invoke(cli_generateScan, [
#         '-c', 'test-cohort',
#         '-s', '1',
#         '-m', 'test-method'])
#     mock_db.assert_called_once()
#     assert "nrrd-e0c2c6182c51052cffc1c4ae4f0e6c1af9a666998f3a291107c060344cf16f64" in mock_db.call_args_list[0][0][1].properties['QualifiedPath']
#     assert result.exit_code == 0 

