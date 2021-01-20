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
