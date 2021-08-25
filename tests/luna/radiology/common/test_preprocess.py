import pytest, os, pathlib
import sys
from medpy.io import load
import numpy as np
from luna.radiology.common.preprocess import *

from luna.common.sparksession import SparkConfig
from luna.common.config import ConfigSet
import luna.common.constants as const
from pathlib import Path
cwd = os.getcwd()

dicom_path = 'tests/luna/testdata/data/2.000000-CTAC-24716/dicoms/1-01.dcm'
image_path = 'tests/luna/testdata/data/2.000000-CTAC-24716/volumes/image.mhd'
label_path = 'tests/luna/testdata/data/2.000000-CTAC-24716/volumes/label.mha'


def test_find_centroid():

    data, header = load(label_path)
    for i in range(data.shape[2]):
        slice = data[:,:,i]
        if np.any(slice):
            image_2d = slice.astype(float).T

            image_2d_scaled = normalize(image_2d)
            image_2d_scaled = np.uint8(image_2d_scaled)

            im = Image.fromarray(image_2d_scaled)

            # save segmentation in red color.
            rgb = im.convert('RGB')
            red_channel = rgb.getdata(0)
            rgb.putdata(red_channel)
            break

    xy = find_centroid(rgb, 512, 512)

    assert 271 == xy[0]
    assert 128 == xy[1]

def test_slice_to_image():

    import luna
    sys.modules['preprocess'] = luna.radiology.common.preprocess

    data, header = load(dicom_path)

    image = slice_to_image(data[:,:,0], 512, 512)

    assert (512, 512) == image.size

def test_subset_bound_seg():

    new_filepath = 'tests/luna/testdata/data/2.000000-CTAC-24716/volumes/subset_image.mhd'
    modified_file = subset_bound_seg(image_path, new_filepath, 0, 3)

    subset_data, subset_hdr = load(modified_file)
    assert 3 == subset_data.shape[2]
    assert 3 == subset_hdr.sitkimage.GetSize()[2]
    
    # cleanup
    os.remove(new_filepath)
    os.remove('tests/luna/testdata/data/2.000000-CTAC-24716/volumes/subset_image.raw')


def test_crop_images():

    ConfigSet(name=const.APP_CFG, config_file='tests/test_config.yml')
    spark = SparkConfig().spark_session(config_name=const.APP_CFG, app_name='test-radiology-utils')

    png = spark.read.format("delta").load("tests/luna/testdata/data/test-project/tables/PNG_dsn").toPandas()
    dicom = png['dicom'][0]
    overlay = png['overlay'][0]
    dicom_overlay = crop_images(360, 198, Image.frombytes("L", (512,512), bytes(dicom)), Image.frombytes("RGB", (512,512), bytes(overlay)), 256, 256, 512, 512)

    assert 256*256 == len(dicom_overlay[0])
    # RGB overlay image
    assert 256*256*3 == len(dicom_overlay[1])


def test_extract_voxels_1(tmp_path):
    properties = extract_voxels(
        image_path = f'tests/luna/testdata/data/2.000000-CTAC-24716/volumes/image.mhd',
        label_path = f'tests/luna/testdata/data/2.000000-CTAC-24716/volumes/label.mha',
        output_dir = str(tmp_path),
        params     = {"resampledPixelSpacing": [2.5,2.5,2.5]}
    )

    print (properties)

    assert properties['targetShape'] == [280, 280, 11] # Check target shape
    assert abs(np.load(str(properties['data']) + '/image_voxels.npy').mean() - -1289.5683001548427) < 1e-8 # Mean within some percision 
    assert np.load(str(properties['data']) + '/label_voxels.npy').sum() == 1139 # ~ number of 1 label voxels


def test_extract_voxels_2(tmp_path):
    properties = extract_voxels(
        image_path = f'tests/luna/testdata/data/2.000000-CTAC-24716/volumes/image.mhd',
        label_path = f'tests/luna/testdata/data/2.000000-CTAC-24716/volumes/label.mha',
        output_dir = str(tmp_path),
        params     = {"resampledPixelSpacing": [5,5,5]}
    )

    print (properties)
    assert properties['targetShape'] == [140, 140, 5] # Check target shape
    assert abs(np.load(str(properties['data']) + '/image_voxels.npy').mean() - -1289.2570235496087) < 1e-8 # Mean within some percision 
    assert np.load(str(properties['data']) + '/label_voxels.npy').sum() == 132 # ~ number of 1 label voxels, less due to different resampling


def test_extract_radiomics_1(tmp_path):
    properties = extract_radiomics(
        image_path = f'tests/luna/testdata/data/2.000000-CTAC-24716/volumes/image.mhd',
        label_path = f'tests/luna/testdata/data/2.000000-CTAC-24716/volumes/label.mha',
        output_dir = str(tmp_path),
        params     = {"job_tag":"test_1", "radiomicsFeatureExtractor": {'interpolator': 'sitkBSpline', 'resampledPixelSpacing': [1, 1, 1], 'padDistance': 10, 'voxelArrayShift': 1000, 'binWidth': 25, 'verbose': 'True', 'label': 1, 'geometryTolerance': 0.0001}}
    )

    assert properties.iloc[0]["original_ngtdm_Strength"].item() == 2.7507247368947003 


def test_extract_radiomics_2(tmp_path):
    properties = extract_radiomics(
        image_path = f'tests/luna/testdata/data/2.000000-CTAC-24716/volumes/image.mhd',
        label_path = f'tests/luna/testdata/data/2.000000-CTAC-24716/volumes/label.mha',
        output_dir = str(tmp_path),
        params     = {"job_tag":"test_1", "radiomicsFeatureExtractor": {'interpolator': 'sitkBSpline', 'resampledPixelSpacing': [1, 1, 1], 'padDistance': 10, 'voxelArrayShift': 1000, 'binWidth': 50, 'verbose': 'True', 'label': 1, 'geometryTolerance': 0.0001}}
    )
    
    assert properties.iloc[0]["original_ngtdm_Strength"].item() ==  0.7069386976494938 


def test_generate_scan_1(tmp_path):
    properties = generate_scan(
        dicom_path = f'tests/luna/testdata/data/2.000000-CTAC-24716/dicoms/',
        output_dir = str(tmp_path),
        params     = {'itkImageType':'mhd'}
    )
    #assert output_node.properties['hash'] == 'eb8574fa61db82aa085ba7c05739d99519b140ca73da95920b887f6bcdba6a9c'
    assert properties['zdim'] == 9
    assert properties['data'] is not None
    assert properties['aux'] is not None


def test_generate_scan_2(tmp_path):
    properties = generate_scan(
        dicom_path = f'tests/luna/testdata/data/2.000000-CTAC-24716/dicoms/',
        output_dir = str(tmp_path),
        params     = {'itkImageType':'nrrd'}
    )
    #assert output_node.properties['hash'] == '53b504fb8fee82e3065104634965fe517cd27c97da97f60057e872c020656262'
    assert properties['zdim'] == 9
    assert properties['data'] is not None


def test_window_dicoms_1(tmp_path):
    properties = window_dicoms(
        dicom_path = f'tests/luna/testdata/data/2.000000-CTAC-24716/dicoms/',
        output_dir = tmp_path,
        params     = {'window':False}
    )
    #assert output_node.properties['hash'] == '05b05657e719f143d68904b1325375c4bd6ad0ee599f3810a5e0e2e5ace4f0bb'
    print (properties)
    assert properties['RescaleSlope'] == 0.0 
    assert properties['RescaleIntercept'] == 0.0 
    assert properties['units'] == 'HU'
    assert os.path.exists(properties['data'])
    assert len(list(properties['data'].glob("*cthu.dcm"))) == 9
    assert np.min(dcmread(str(properties['data']) + '/1-05.cthu.dcm').pixel_array) == -3024
    assert np.max(dcmread(str(properties['data']) + '/1-05.cthu.dcm').pixel_array) ==  1387

def test_window_dicoms_2(tmp_path):
    properties = window_dicoms(
        dicom_path = f'tests/luna/testdata/data/2.000000-CTAC-24716/dicoms/',
        output_dir = tmp_path,
        params     = {'window':True, 'windowLowLevel': -100, 'windowHighLevel': 100}
    )
    #assert output_node.properties['hash'] == '5624a11d08ab8ef4e66e3fd9307e775bf8bbad7d0759aab893d1648d7c60ae19'
    print (properties)
    assert properties['RescaleSlope'] == 0.0 
    assert properties['RescaleIntercept'] == 0.0
    assert properties['units'] == 'HU'
    assert os.path.exists(properties['data'])
    assert len(list(properties['data'].glob("*cthu.dcm"))) == 9
    assert np.min(dcmread(str(properties['data']) + '/1-05.cthu.dcm').pixel_array) == -100
    assert np.max(dcmread(str(properties['data']) + '/1-05.cthu.dcm').pixel_array) ==  100



