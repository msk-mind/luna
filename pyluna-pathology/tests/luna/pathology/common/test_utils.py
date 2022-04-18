import pytest

from luna.common.config import ConfigSet
import luna.common.constants as const
from luna.pathology.common.utils import get_labelset_keys, convert_xml_to_mask, convert_halo_xml_to_roi
import numpy as np


from luna.pathology.common.utils import *

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


def test_get_labelset_keys():

    cfg = ConfigSet(name=const.DATA_CFG,
                    config_file='pyluna-pathology/tests/luna/pathology/common/testdata/point_geojson_config.yaml')

    res = get_labelset_keys()

    assert 1 == len(res)
    assert 'LYMPHOCYTE_DETECTION_LABELSET' == res[0]


xml_data_path = 'pyluna-pathology/tests/luna/pathology/testdata/data/test-project/pathology_annotations/123456_annotation_from_halo.xml'

def test_convert_halo_xml_to_roi():
   roi = convert_halo_xml_to_roi(xml_data_path)
   assert roi == ([1, 10], [40, 1])

def test_convert_xml_to_mask():
   roi = convert_xml_to_mask(xml_data_path, shape=(20,50), annotation_name="Tumor")
   assert list(np.bincount(roi.flatten())) == [929,  71]
   assert list(roi[5,:7]) == [False, False,  True,  True, True,  True, False]

def test_get_tile_array():
    import pandas as pd
    df = pd.DataFrame({'address':['x4_y6_z20.0'],
                       'tile_store':'pyluna-pathology/tests/luna/pathology/common/testdata/123.tiles.h5'}) \
        .set_index('address')

    tile_arr = get_tile_array(df.loc['x4_y6_z20.0'])
    assert np.ndarray == type(tile_arr)
    assert (256, 256, 3) == tile_arr.shape

