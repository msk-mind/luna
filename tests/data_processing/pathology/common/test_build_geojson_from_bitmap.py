from data_processing.pathology.common.build_geojson_from_bitmap import *
import os
import numpy as np

npy_data_path = 'tests/data_processing/testdata/data/test-project/pathology_annotations/regional_npys/2021_HobS21_8_123'

def test_concatenate_geojsons_from_list_with_one_geojson():
    geojson_list = [{"type":"FeatureCollection","features":[{"type":"Feature","properties":{"label_num":"1","label_name":"tissue_1"},"geometry":{"type":"Polygon","coordinates":[[1261,2140],[1236,2140],[1222,2134],[1222,2132],[1216,2125]]}}]}]

    concat_geojson = concatenate_geojsons_from_list(geojson_list)

    assert 1 == len(concat_geojson['features'])

def test_concatenate_geojsons_from_list():
    geojson_list = [{"type":"FeatureCollection","features":[{"type":"Feature","properties":{"label_num":"1","label_name":"tissue_1"},"geometry":{"type":"Polygon","coordinates":[[1261,2140],[1236,2140],[1222,2134],[1222,2132],[1216,2125]]}}]},
                    {"type":"FeatureCollection","features":[{"type":"Feature","properties":{"label_num":"3","label_name":"tissue_3"},"geometry":{"type":"Polygon","coordinates":[[1261,2140],[1236,2140],[1222,2134],[1222,2132],[1216,2125]]}}]}]
    concat_geojson = concatenate_geojsons_from_list(geojson_list)

    assert isinstance(concat_geojson, dict)
    assert 2 == len(concat_geojson['features'])

def test_add_contours_for_label():
    base_geojson = {
        "type": "FeatureCollection",
        "features": []
    }
    annotation = np.load(os.path.join(npy_data_path, '123_jill_SVBMP-123sdf_annot.npy'))
    res = add_contours_for_label(base_geojson, annotation, 1, {1:"tissue_1"}, 0.5, 1)

    assert 1 < len(res['features'][0]['geometry']['coordinates'])

def test_add_contours_for_label_non_matching_label():
    base_geojson = {
        "type": "FeatureCollection",
        "features": []
    }
    annotation = np.load(os.path.join(npy_data_path, '123_jill_SVBMP-123sdf_annot.npy'))
    res = add_contours_for_label(base_geojson, annotation, 3, {1:"tissue_1"}, 0.5, 1)

    assert base_geojson == res


def test_build_geojson_from_bitmap():

    res = build_geojson_from_bitmap('tests/data_processing/pathology/test_regional_etl_config.yaml',
                                    'test-project',
                                    os.path.join(npy_data_path, '123_joe_SVBMP-123asd_annot.npy'),
                                    'default_labels',
                                    0.5,
                                    1)
    assert isinstance(res, dict)
    assert 1 == len(res['features'])


def test_concatenate_regional_geojsons():

    geojson_list = ['{"type":"FeatureCollection","features":[{"type":"Feature","properties":{"label_num":"1","label_name":"tissue_1"},"geometry":{"type":"Polygon","coordinates":[[1261,2140],[1236,2140],[1222,2134],[1222,2132],[1216,2125]]}}]}',
                    '{"type":"FeatureCollection","features":[{"type":"Feature","properties":{"label_num":"3","label_name":"tissue_3"},"geometry":{"type":"Polygon","coordinates":[[1117,844],[1074,844],[1062,836],[1056,830],[1054,825]]}}]}']

    res = concatenate_regional_geojsons(geojson_list)

    assert isinstance(res, dict)
    assert 2 == len(res['features'])
