from data_processing.common.config import ConfigSet
import data_processing.common.constants as const
from data_processing.pathology.common.utils import get_labelset_keys


def test_get_labelset_keys():

    cfg = ConfigSet(name=const.DATA_CFG,
                    config_file='tests/data_processing/pathology/common/testdata/point_geojson_config.yaml')

    res = get_labelset_keys()

    assert 1 == len(res)
    assert 'LYMPHOCYTE_DETECTION_LABELSET' == res[0]
