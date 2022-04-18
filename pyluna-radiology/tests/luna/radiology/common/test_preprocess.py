import os
import sys
from medpy.io import load
import numpy as np
from luna.radiology.common.preprocess import *

from luna.common.config import ConfigSet
import luna.common.constants as const
from pathlib import Path
cwd = os.getcwd()

data_dir = 'pyluna-radiology/tests/luna/testdata/data'
dicom_path = f'{data_dir}/2.000000-CTAC-24716/dicoms/1-01.dcm'
image_path = f'{data_dir}/2.000000-CTAC-24716/volumes/image.mhd'
label_path = f'{data_dir}/2.000000-CTAC-24716/volumes/label.mha'

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

    new_filepath = f'{data_dir}/2.000000-CTAC-24716/volumes/subset_image.mhd'
    modified_file = subset_bound_seg(image_path, new_filepath, 0, 3)

    subset_data, subset_hdr = load(modified_file)
    assert 3 == subset_data.shape[2]
    assert 3 == subset_hdr.sitkimage.GetSize()[2]
    
    # cleanup
    os.remove(new_filepath)
    os.remove(f'{data_dir}/2.000000-CTAC-24716/volumes/subset_image.raw')


# def test_crop_images():
# 
#     ConfigSet(name=const.APP_CFG, config_file='pyluna-radiology/tests/test_config.yml')
#     spark = SparkConfig().spark_session(config_name=const.APP_CFG, app_name='test-radiology-utils')
# 
#     png = spark.read.format("delta").load(f"{data_dir}/test-project/tables/PNG_dsn").toPandas()
#     dicom = png['dicom'][0]
#     overlay = png['overlay'][0]
#     dicom_overlay = crop_images(360, 198, Image.frombytes("L", (512,512), bytes(dicom)), Image.frombytes("RGB", (512,512), bytes(overlay)), 256, 256, 512, 512)
# 
#     assert 256*256 == len(dicom_overlay[0])
#     # RGB overlay image
#     assert 256*256*3 == len(dicom_overlay[1])
