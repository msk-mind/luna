import os, sys
import numpy as np
import pandas as pd
import json
from luna.pathology.common.build_geojson import *

npy_data_path = 'pyluna-pathology/tests/luna/pathology/testdata/data/test-project/pathology_annotations/regional_npys/2021_HobS21_8_123'

def test_add_contours_for_label():
    base_geojson = {
        "type": "FeatureCollection",
        "features": []
    }
    annotation = np.load(os.path.join(npy_data_path, '123_jill_SVBMP-123sdf_annot.npy'))
    res = add_contours_for_label(base_geojson, annotation, 1, {1:"tissue_1"}, 0.5)

    assert 1 < len(res['features'][0]['geometry']['coordinates'][0])

def test_add_contours_for_label_non_matching_label():
    base_geojson = {
        "type": "FeatureCollection",
        "features": []
    }
    annotation = np.load(os.path.join(npy_data_path, '123_jill_SVBMP-123sdf_annot.npy'))
    res = add_contours_for_label(base_geojson, annotation, 3, {1:"tissue_1"}, 0.5)

    assert base_geojson == res


def test_build_geojson_from_bitmap():
    import luna
    sys.modules['build_geojson'] = luna.pathology.common.build_geojson

    data = [{"label_config": "{'DEFAULT_LABELS': {1: 'tissue_1', 2: 'tissue_2', 3: 'tissue_3', 4: 'tissue_4', 5: 'tissue_5'}}",
             "npy_filepath": os.path.join(npy_data_path, '123_joe_SVBMP-123asd_annot.npy'),
             "labelset": "DEFAULT_LABELS",
             "contour_level": 0.5,
             "geojson": ""}]

    df = pd.DataFrame(data)
    res = build_geojson_from_annotation(df)

    # get geojson column
    geojson = res.geojson.item()
    geojson_dict = json.loads(geojson)

    assert isinstance(geojson, str)
    assert 1 == len(geojson_dict['features'])


def test_concatenate_regional_geojsons():

    geojson_list = ['{"type":"FeatureCollection","features":[{"type":"Feature","properties":{"label_num":"1","label_name":"tissue_1"},"geometry":{"type":"Polygon","coordinates":[[1261,2140],[1236,2140],[1222,2134],[1222,2132],[1216,2125]]}}]}',
                    '{"type":"FeatureCollection","features":[{"type":"Feature","properties":{"label_num":"3","label_name":"tissue_3"},"geometry":{"type":"Polygon","coordinates":[[1117,844],[1074,844],[1062,836],[1056,830],[1054,825]]}}]}']

    res = concatenate_regional_geojsons(geojson_list)

    assert isinstance(res, dict)
    assert 2 == len(res['features'])

def test_concatenate_regional_geojsons_one_geojson():

    geojson_list = ['{"type":"FeatureCollection","features":[{"type":"Feature","properties":{"label_num":"1","label_name":"tissue_1"},"geometry":{"type":"Polygon","coordinates":[[1261,2140],[1236,2140],[1222,2134],[1222,2132],[1216,2125]]}}]}']
    res = concatenate_regional_geojsons(geojson_list)

    assert isinstance(res, dict)
    assert 1 == len(res['features'])

def test_build_geojson_from_pointclick_json():

    res = build_geojson_from_pointclick_json("{'DEFAULT_LABELS': {0: 'tissue_1', 2: 'tissue_2', 3: 'tissue_3', 4: 'tissue_4', 5: 'tissue_5'}}",
                                             "DEFAULT_LABELS",
                                             [{"project_id":"8","image_id":"123.svs","label_type":"nucleus","x":"1440","y":"747","class":"0","classname":"tissue_1"},
                                              {"project_id":"8","image_id":"123.svs","label_type":"nucleus","x":"1040","y":"477","class":"3","classname":"tissue_3"}])

    assert 2 == len(res)
    assert "Feature" == res[0]['type']
    assert "PathAnnotationObject" == res[0]['id']
    assert isinstance(res[0]['properties'], dict)
    assert 2 == len(res[0]['geometry']['coordinates'])
