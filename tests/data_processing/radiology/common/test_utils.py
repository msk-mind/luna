import pytest, os, pathlib
from data_processing.radiology.common.utils import *
from data_processing.common.sparksession import SparkConfig
from data_processing.common.config import ConfigSet
import data_processing.common.constants as const

cwd = os.getcwd()

dicom_path = f'{cwd}/tests/data_processing/testdata/data/2.000000-CTAC-24716/dicoms/1-01.dcm'
image_path = f'{cwd}/tests/data_processing/testdata/data/2.000000-CTAC-24716/volumes/image.mhd'
label_path = f'{cwd}/tests/data_processing/testdata/data/2.000000-CTAC-24716/volumes/label.mha'


def test_find_centroid():

    xy = find_centroid(label_path, 512, 512)
    assert 271 == xy[0]
    assert 128 == xy[1]

def test_dicom_to_bytes():

    image = dicom_to_bytes(dicom_path, 512, 512)

    assert bytes == type(image)
    assert 512*512 == len(image)

def test_create_seg_images():

    arr = create_seg_images(image_path, "uuid", 512, 512)

    assert 9 == len(arr)
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

def test_extract_radiomics_1(tmp_path):
    output_node = extract_radiomics(
        name = "test_radiomics_1",
        image_path = f'{cwd}/tests/data_processing/testdata/data/2.000000-CTAC-24716/volumes/image.mhd',
        label_path = f'{cwd}/tests/data_processing/testdata/data/2.000000-CTAC-24716/volumes/label.mha',
        output_dir = tmp_path,
        params     = {"RadiomicsFeatureExtractor": {'interpolator': 'sitkBSpline', 'resampledPixelSpacing': [1, 1, 1], 'padDistance': 10, 'voxelArrayShift': 1000, 'binWidth': 25, 'verbose': 'True', 'label': 1, 'geometryTolerance': 0.0001}}
    )

    assert output_node.properties['hash'] == 'ae5665617279299e1d4bc8a3e2434f4a4616c4a096ba4ab47d41f74cecea2103'
    assert output_node.properties['qualified_address'] == 'default::test_radiomics_1'
    assert output_node.name == 'test_radiomics_1'
    assert output_node.type == 'radiomics'


def test_extract_radiomics_2(tmp_path):
    output_node = extract_radiomics(
        name = "test_radiomics_2",
        image_path = f'{cwd}/tests/data_processing/testdata/data/2.000000-CTAC-24716/volumes/image.mhd',
        label_path = f'{cwd}/tests/data_processing/testdata/data/2.000000-CTAC-24716/volumes/label.mha',
        output_dir = tmp_path,
        params     = {"RadiomicsFeatureExtractor": {'interpolator': 'sitkBSpline', 'resampledPixelSpacing': [1, 1, 1], 'padDistance': 10, 'voxelArrayShift': 1000, 'binWidth': 50, 'verbose': 'True', 'label': 1, 'geometryTolerance': 0.0001}}
    )

    assert output_node.properties['hash'] == 'b1cf9121d8ed9d8c788a10d837f0c7799124cef08223093c135034ff3370f8a7'
    assert output_node.properties['qualified_address'] == 'default::test_radiomics_2'
    assert output_node.name == 'test_radiomics_2'
    assert output_node.type == 'radiomics'



def test_generate_scan_1(tmp_path):
    output_node = generate_scan(
        name = "test_generate_scan_1",
        dicom_path = f'{cwd}/tests/data_processing/testdata/data/2.000000-CTAC-24716/dicoms/',
        output_dir = tmp_path,
        params     = {'file_ext':'mhd'}
    )
    assert output_node.properties['hash'] == 'eb8574fa61db82aa085ba7c05739d99519b140ca73da95920b887f6bcdba6a9c'
    assert output_node.properties['qualified_address'] == 'default::test_generate_scan_1'
    assert output_node.name == 'test_generate_scan_1'
    assert output_node.type == 'mhd'


def test_generate_scan_2(tmp_path):
    output_node = generate_scan(
        name = "test_generate_scan_2",
        dicom_path = f'{cwd}/tests/data_processing/testdata/data/2.000000-CTAC-24716/dicoms/',
        output_dir = tmp_path,
        params     = {'file_ext':'nrrd'}
    )
    assert output_node.properties['hash'] == '53b504fb8fee82e3065104634965fe517cd27c97da97f60057e872c020656262'
    assert output_node.properties['qualified_address'] == 'default::test_generate_scan_2'
    assert output_node.name == 'test_generate_scan_2'
    assert output_node.type == 'nrrd'



def test_window_dicoms_1(tmp_path):
    output_node = window_dicoms(
        name = "test_window_dicoms_1",
        dicom_paths = list(pathlib.Path(f'{cwd}/tests/data_processing/testdata/data/2.000000-CTAC-24716/dicoms/').glob("*.dcm")),
        output_dir = tmp_path,
        params     = {'window':False}
    )
    assert output_node.properties['hash'] == '05b05657e719f143d68904b1325375c4bd6ad0ee599f3810a5e0e2e5ace4f0bb'
    assert output_node.properties['qualified_address'] == 'default::test_window_dicoms_1'
    assert output_node.name == 'test_window_dicoms_1'
    assert output_node.type == 'dicom'



def test_window_dicoms_2(tmp_path):
    output_node = window_dicoms(
        name = "test_window_dicoms_2",
        dicom_paths = list(pathlib.Path(f'{cwd}/tests/data_processing/testdata/data/2.000000-CTAC-24716/dicoms/').glob("*.dcm")),
        output_dir = tmp_path,
        params     = {'window':True, 'window.low_level': -100, 'window.high_level': 100}
    )
    assert output_node.properties['hash'] == '5624a11d08ab8ef4e66e3fd9307e775bf8bbad7d0759aab893d1648d7c60ae19'
    assert output_node.properties['qualified_address'] == 'default::test_window_dicoms_2'
    assert output_node.name == 'test_window_dicoms_2'
    assert output_node.type == 'dicom'


