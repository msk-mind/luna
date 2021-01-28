from data_processing.pathology.common.build_geojson_from_bitmap import *
import os
import numpy as np
import pandas as pd

geojson_data_path = 'tests/data_processing/testdata/data/test-project/pathology_annotations/regional_geojsons/default_labels'
npy_data_path = 'tests/data_processing/testdata/data/test-project/pathology_annotations/regional_npys/2021_HobS21_8_123'

def test_concatenate_geojsons_from_list():
    geojson_list = [os.path.join(geojson_data_path, '2021_HobS21_8_123/123_jill_SVBMP-123sdf_annot_geojson.json'),
                    os.path.join(geojson_data_path, '2021_HobS21_8_123/123_joe_SVBMP-123asd_annot_geojson.json')]

    concat_geojson = concatenate_geojsons_from_list(geojson_list)

    assert 2 == len(concat_geojson['features'])

def test_concatenate_geojsons_from_list_with_one_geojson():
    geojson_list = [os.path.join(geojson_data_path, '2021_HobS21_8_123/123_jill_SVBMP-123sdf_annot_geojson.json')]

    concat_geojson = concatenate_geojsons_from_list(geojson_list)

    assert 1 == len(concat_geojson['features'])

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


def test_build_geojson_from_bitmap_pandas():

    data = [('test-project',
             'tests/data_processing/pathology/test_regional_etl_config.yaml',
             'default_labels',
             os.path.join(npy_data_path, '123_jill_SVBMP-123sdf_annot.npy'),
             geojson_data_path,
             0.5,
             1,
             '',
             '')]
    columns = ['dmt','configuration_file','labelset','npy_filepath','slide_json_dir','contour_level','polygon_tolerance','geojson_filepath','geojson_record_uuid']
    df = pd.DataFrame(data, columns=columns)

    res = build_geojson_from_bitmap_pandas(df)

    assert os.path.join(geojson_data_path, '2021_HobS21_8_123/123_jill_SVBMP-123sdf_annot_geojson.json') == res.geojson_filepath.values[0]
    assert res.geojson_record_uuid.values[0].startswith('SVGEOJSON-default_labels-')


def test_concatenate_regional_geojsons_pandas():

    concat_geojson_path = 'tests/data_processing/testdata/data/test-project/pathology_annotations/regional_concat_geojsons'
    data = [('tests/data_processing/pathology/test_regional_etl_config.yaml',
             'default_labels',
             '123',
             '2021;HobS21;8;123.svs',
             concat_geojson_path,
             0.5,
             1,
             os.path.join(geojson_data_path, '2021_HobS21_8_123/123_jill_SVBMP-123sdf_annot_geojson.json'),
             'SVGEOJSON-default_labels-a65a9ab7f8458668439568d5a6c8ab3dd1605280cf42c89e8c4122d646a70d2f',
             '',
             '',
             'jill'),
            ('tests/data_processing/pathology/test_regional_etl_config.yaml',
             'default_labels',
             '123',
             '2021;HobS21;8;123.svs',
             concat_geojson_path,
             0.5,
             1,
             os.path.join(geojson_data_path, '2021_HobS21_8_123/123_joe_SVBMP-123asd_annot_geojson.json'),
             'SVGEOJSON-default_labels-adfg',
             '',
             '',
             'joe')]
    columns = ['configuration_file','labelset','slide_id','slideviewer_path','slide_json_dir','contour_level','polygon_tolerance','geojson_filepath','geojson_record_uuid',
               'concat_geojson_filepath','concat_geojson_record_uuid','user']
    df = pd.DataFrame(data, columns=columns)

    res = concatenate_regional_geojsons_pandas(df)

    assert 2 == len(res)
    assert os.path.join(concat_geojson_path,
                        '2021_HobS21_8_123/123_annot_concat_geojson.json') == res.concat_geojson_filepath.values[0]
    assert 'CONCAT' == res.user.values[0]
    assert 'joe' == res.user.values[1] # gets purged in generate.py
    assert res.concat_geojson_record_uuid.values[0].startswith('SVCONCATGEOJSON-default_labels-')
