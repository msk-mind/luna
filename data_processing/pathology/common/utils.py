from data_processing.common.config import ConfigSet
import data_processing.common.constants as const

def get_labelset_keys():
    """
    Given DATA_CFG, return slideviewer labelsets

    :return: list of labelset names
    """
    cfg = ConfigSet()
    label_config = cfg.get_value(path=const.DATA_CFG+'::LABEL_SETS')
    labelsets = [cfg.get_value(path=const.DATA_CFG+'::USE_LABELSET')]

    if cfg.get_value(path=const.DATA_CFG+'::USE_ALL_LABELSETS'):
        labelsets = list(label_config.keys())

    return labelsets
