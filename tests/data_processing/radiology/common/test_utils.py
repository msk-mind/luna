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

def test_radiomics_1(tmp_path):
    output_node = extract_radiomics(
        name = "test_radiomics_1",
        image_path = f'{cwd}/tests/data_processing/testdata/data/2.000000-CTAC-24716/volumes/image.mhd',
        label_path = f'{cwd}/tests/data_processing/testdata/data/2.000000-CTAC-24716/volumes/label.mha',
        output_dir = tmp_path,
        params     = {"RadiomicsFeatureExtractor": {'interpolator': 'sitkBSpline', 'resampledPixelSpacing': [1, 1, 1], 'padDistance': 10, 'voxelArrayShift': 1000, 'binWidth': 25, 'verbose': 'True', 'label': 1, 'geometryTolerance': 0.0001}}
    )

    assert "3227.840849331449,0.09075042902243616,2.7507247368947003\n" in open(str(output_node.properties['path']) + '/radiomics-out.csv').read()

    assert output_node.properties['hash'] == '7139f9e9ad823ddc581a5d4b61f31be2e60d4ba857557a9dfbb6425f9ef567b8'
    assert output_node.properties['qualified_address'] == 'default::test_radiomics_1'
    assert output_node.name == 'test_radiomics_1'
    assert output_node.type == 'radiomics'


def test_radiomics_2(tmp_path):
    output_node = extract_radiomics(
        name = "test_radiomics_2",
        image_path = f'{cwd}/tests/data_processing/testdata/data/2.000000-CTAC-24716/volumes/image.mhd',
        label_path = f'{cwd}/tests/data_processing/testdata/data/2.000000-CTAC-24716/volumes/label.mha',
        output_dir = tmp_path,
        params     = {"RadiomicsFeatureExtractor": {'interpolator': 'sitkBSpline', 'resampledPixelSpacing': [1, 1, 1], 'padDistance': 10, 'voxelArrayShift': 1000, 'binWidth': 50, 'verbose': 'True', 'label': 1, 'geometryTolerance': 0.0001}}
    )
    print (str(output_node.properties['path']) + '/radiomics-out.csv')
    assert ",0.001316830812757558,447.00957648375726,0.04525463261369965,0.7069386976494938\n" in open(str(output_node.properties['path']) + '/radiomics-out.csv').read()

    assert output_node.properties['hash'] == '1c57aa6db3ab3251c9ef65b826348e599003e395978a1287dcd248f2249b7a0b'
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
    assert output_node.properties['hash'] == 'accbcd2d504d0889c59e306f73219d3c2bbeeb362f09414712f81dbdc47efc7a'
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
    assert output_node.properties['hash'] == 'a1354c7b965e1528d6aefd1b4bf3783743f57feb4f24c19cf076f15e823779d1'
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
    assert output_node.properties['hash'] == '44ee34edb94ca28c48693d71b34a52c86ebcfd9ed512b281dfe7666391b1e82d'
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
    assert output_node.properties['hash'] == 'dcfc9d1ae40651964aa49b49371f1f94d40aaf01ffa561b4f897687fc0d026ff'
    assert output_node.properties['qualified_address'] == 'default::test_window_dicoms_2'
    assert output_node.name == 'test_window_dicoms_2'
    assert output_node.type == 'dicom'


