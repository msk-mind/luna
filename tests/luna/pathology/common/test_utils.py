from luna.common.config import ConfigSet
import luna.common.constants as const
from luna.pathology.common.utils import get_labelset_keys, convert_xml_to_mask, convert_halo_xml_to_roi
import numpy as np


def test_get_labelset_keys():

    cfg = ConfigSet(name=const.DATA_CFG,
                    config_file='tests/luna/pathology/common/testdata/point_geojson_config.yaml')

    res = get_labelset_keys()

    assert 1 == len(res)
    assert 'LYMPHOCYTE_DETECTION_LABELSET' == res[0]


xml_data_path = 'tests/luna/pathology/testdata/data/test-project/pathology_annotations/123456_annotation_from_halo.xml'

def test_convert_halo_xml_to_roi():
   roi = convert_halo_xml_to_roi(xml_data_path)
   assert roi == ([1, 10], [40, 1])

def test_convert_xml_to_mask():
   roi = convert_xml_to_mask(xml_data_path, shape=(20,50), annotation_name="Tumor")
   assert list(np.bincount(roi.flatten())) == [929,  71]
   assert list(roi[5,:7]) == [False, False,  True,  True, True,  True, False]
