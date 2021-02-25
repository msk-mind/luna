import pytest, os, pathlib
import sys
from data_processing.radiology.common.preprocess import *

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

    import data_processing
    sys.modules['preprocess'] = data_processing.radiology.common.preprocess

    image = dicom_to_bytes(dicom_path, 512, 512)

    assert bytes == type(image)
    assert 512*512 == len(image)

def test_create_seg_images():

    import data_processing
    sys.modules['preprocess'] = data_processing.radiology.common.preprocess

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


def test_extract_voxels_1(tmp_path):
    properties = extract_voxels(
        image_path = f'{cwd}/tests/data_processing/testdata/data/2.000000-CTAC-24716/volumes/image.mhd',
        label_path = f'{cwd}/tests/data_processing/testdata/data/2.000000-CTAC-24716/volumes/label.mha',
        output_dir = tmp_path,
        params     = {"resampledPixelSpacing": [2.5,2.5,2.5]}
    )

    print (properties)

    assert properties['targetShape'] == [280, 280, 11] # Check target shape
    assert abs(np.load(str(properties['path']) + '/image_voxels.npy').mean() - -1289.5683001548427) < 1e-8 # Mean within some percision 
    assert np.load(str(properties['path']) + '/label_voxels.npy').sum() == 1139 # ~ number of 1 label voxels


def test_extract_voxels_2(tmp_path):
    properties = extract_voxels(
        image_path = f'{cwd}/tests/data_processing/testdata/data/2.000000-CTAC-24716/volumes/image.mhd',
        label_path = f'{cwd}/tests/data_processing/testdata/data/2.000000-CTAC-24716/volumes/label.mha',
        output_dir = tmp_path,
        params     = {"resampledPixelSpacing": [5,5,5]}
    )

    print (properties)
    assert properties['targetShape'] == [140, 140, 5] # Check target shape
    assert abs(np.load(str(properties['path']) + '/image_voxels.npy').mean() - -1289.2570235496087) < 1e-8 # Mean within some percision 
    assert np.load(str(properties['path']) + '/label_voxels.npy').sum() == 132 # ~ number of 1 label voxels, less due to different resampling


def test_extract_radiomics_1(tmp_path):
    properties = extract_radiomics(
        image_path = f'{cwd}/tests/data_processing/testdata/data/2.000000-CTAC-24716/volumes/image.mhd',
        label_path = f'{cwd}/tests/data_processing/testdata/data/2.000000-CTAC-24716/volumes/label.mha',
        output_dir = tmp_path,
        params     = {"RadiomicsFeatureExtractor": {'interpolator': 'sitkBSpline', 'resampledPixelSpacing': [1, 1, 1], 'padDistance': 10, 'voxelArrayShift': 1000, 'binWidth': 25, 'verbose': 'True', 'label': 1, 'geometryTolerance': 0.0001}}
    )

    assert "3227.840849331449,0.09075042902243616,2.7507247368947003\n" in open(str(properties['path']) + '/radiomics-out.csv').read() # Check the last


def test_extract_radiomics_2(tmp_path):
    properties = extract_radiomics(
        image_path = f'{cwd}/tests/data_processing/testdata/data/2.000000-CTAC-24716/volumes/image.mhd',
        label_path = f'{cwd}/tests/data_processing/testdata/data/2.000000-CTAC-24716/volumes/label.mha',
        output_dir = tmp_path,
        params     = {"RadiomicsFeatureExtractor": {'interpolator': 'sitkBSpline', 'resampledPixelSpacing': [1, 1, 1], 'padDistance': 10, 'voxelArrayShift': 1000, 'binWidth': 50, 'verbose': 'True', 'label': 1, 'geometryTolerance': 0.0001}}
    )

    assert "0.001316830812757558,447.00957648375726,0.04525463261369965,0.7069386976494938\n" in open(str(properties['path']) + '/radiomics-out.csv').read()


def test_generate_scan_1(tmp_path):
    properties = generate_scan(
        dicom_path = f'{cwd}/tests/data_processing/testdata/data/2.000000-CTAC-24716/dicoms/',
        output_dir = tmp_path,
        params     = {'file_ext':'mhd'}
    )
    #assert output_node.properties['hash'] == 'eb8574fa61db82aa085ba7c05739d99519b140ca73da95920b887f6bcdba6a9c'
    assert properties['zdim'] == 9
    assert len(list(properties['path'].glob("*"))) == 2


def test_generate_scan_2(tmp_path):
    properties = generate_scan(
        dicom_path = f'{cwd}/tests/data_processing/testdata/data/2.000000-CTAC-24716/dicoms/',
        output_dir = tmp_path,
        params     = {'file_ext':'nrrd'}
    )
    #assert output_node.properties['hash'] == '53b504fb8fee82e3065104634965fe517cd27c97da97f60057e872c020656262'
    assert properties['zdim'] == 9
    assert len(list(properties['path'].glob("*"))) == 1


def test_window_dicoms_1(tmp_path):
    properties = window_dicoms(
        dicom_paths = list(pathlib.Path(f'{cwd}/tests/data_processing/testdata/data/2.000000-CTAC-24716/dicoms/').glob("*.dcm")),
        output_dir = tmp_path,
        params     = {'window':False}
    )
    #assert output_node.properties['hash'] == '05b05657e719f143d68904b1325375c4bd6ad0ee599f3810a5e0e2e5ace4f0bb'
    print (properties)
    assert properties['RescaleSlope'] == 1.0
    assert properties['RescaleIntercept'] == -1024.0
    assert properties['units'] == 'HU'
    assert os.path.exists(properties['path'])
    assert len(list(properties['path'].glob("*cthu.dcm"))) == 9
    assert np.min(dcmread(str(properties['path']) + '/1-05.cthu.dcm').pixel_array) == -3024
    assert np.max(dcmread(str(properties['path']) + '/1-05.cthu.dcm').pixel_array) ==  1387

def test_window_dicoms_2(tmp_path):
    properties = window_dicoms(
        dicom_paths = list(pathlib.Path(f'{cwd}/tests/data_processing/testdata/data/2.000000-CTAC-24716/dicoms/').glob("*.dcm")),
        output_dir = tmp_path,
        params     = {'window':True, 'window.low_level': -100, 'window.high_level': 100}
    )
    #assert output_node.properties['hash'] == '5624a11d08ab8ef4e66e3fd9307e775bf8bbad7d0759aab893d1648d7c60ae19'
    print (properties)
    assert properties['RescaleSlope'] == 1.0
    assert properties['RescaleIntercept'] == -1024.0
    assert properties['units'] == 'HU'
    assert os.path.exists(properties['path'])
    assert len(list(properties['path'].glob("*cthu.dcm"))) == 9
    assert np.min(dcmread(str(properties['path']) + '/1-05.cthu.dcm').pixel_array) == -100
    assert np.max(dcmread(str(properties['path']) + '/1-05.cthu.dcm').pixel_array) ==  100



