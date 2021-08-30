import pytest
import os, shutil
import json
import numpy as np
import openslide
from openslide.deepzoom import DeepZoomGenerator

from luna.pathology.common.preprocess import *

output_dir = "pyluna-pathology/tests/luna/pathology/common/testdata/output-123"
slide_path = "pyluna-pathology/tests/luna/pathology/common/testdata/123.svs"
scores_csv_path = "pyluna-pathology/tests/luna/pathology/common/testdata/input/tile_scores_and_labels.csv"
slide = openslide.OpenSlide(slide_path)
img_arr = get_downscaled_thumbnail(slide, 10)

def test_get_scale_factor_at_magnification_double():
    # slide scanned mag is 20
    res = get_scale_factor_at_magnfication(slide, 10)
    assert 2 == res

def test_get_scale_factor_at_magnification_error():
    # slide scanned mag is 20
    with pytest.raises(ValueError):
        get_scale_factor_at_magnfication(slide, 40)

def test_get_tile_color():

    res = get_tile_color(0.1)
    assert 3 == len(res)

def test_get_tile_color_str():

    res = get_tile_color("blue")
    assert 3 == len(res)

def test_get_full_resolution_generator():

    generator, level = get_full_resolution_generator(slide, 128)
    assert isinstance(generator, DeepZoomGenerator)
    assert 12 == level

def test_get_downscaled_thumbnail():
    res = get_downscaled_thumbnail(slide, 10)

    assert isinstance(res, np.ndarray)

def test_array_to_slide():
    res = array_to_slide(img_arr)

    assert isinstance(res, openslide.ImageSlide)

def test_make_otsu():

    res = make_otsu(img_arr[:,:,1])

    assert 0 == np.count_nonzero(res[0])
    assert isinstance(res, np.ndarray)

def test_pretile_scoring(requests_mock):

    # setup
    os.makedirs(output_dir, exist_ok=True)

    params = {"tile_size":128,
              "requested_magnification":20,
              "project_id": "project",
              "labelset": "default_labels",
              "filter": {
                  "otsu_score": 0.5
              },
              "annotation_table_path": "pyluna-pathology/tests/luna/pathology/common/testdata/project/tables/REGIONAL_METADATA_RESULTS"
              }
    res = pretile_scoring(slide_path, output_dir,
                          "pyluna-pathology/tests/luna/pathology/common/testdata/project/tables/REGIONAL_METADATA_RESULTS",
                          params, "123")

    print(res)
    assert 'pyluna-pathology/tests/luna/pathology/common/testdata/output-123/tiles.slice.pil' == res['data']
    assert 'pyluna-pathology/tests/luna/pathology/common/testdata/output-123/address.slice.csv' == res['aux']
    assert 'RGB' == res['pil_image_bytes_mode']
    assert 20 == res['full_resolution_magnification']
    assert ['coordinates', 'otsu_score', 'purple_score', 'regional_label'] == res['available_labels']
    assert '123.svs' == res['image_filename']

    # clean up
    shutil.rmtree(output_dir)

"""
# works on a cuda enabled env
def test_run_model():

    params = {
        "model_package": "luna.pathology.models.eng_tissuenet",
        "model": {
            "checkpoint_path": "/gpfs/mskmindhdp_emc/user/shared_data_folder/kohlia/tile_classifier/ckpts/4.ckpt",
            "n_classes": 5
        }
    }
    res = run_model('/gpfs/mskmindhdp_emc/data/TCGA-BRCA/TCGA-D8-A4Z1-01Z-00-DX1.D39D38B5-FC9F-4298-8720-016407DC6591/test_collect_tiles/tiles.slice.pil',
                    '/gpfs/mskmindhdp_emc/data/TCGA-BRCA/TCGA-D8-A4Z1-01Z-00-DX1.D39D38B5-FC9F-4298-8720-016407DC6591/test_collect_tiles/address.slice.csv',
                    'pyluna-pathology/tests/luna/pathology/common/testdata', params)

    print(res)
"""
