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


def get_add_triple_str(slide_id, type, record_uuid):
    """
    Populates add cypher string.

    :param slide_id: pathology slide id
    :param type: type of the node e.g. bmp_record_uuid
    :param record_uuid: record_uuid value
    :return: query string
    """
    add_str = f'''MERGE (n:slide{{slide_id: "{slide_id}"}}) MERGE (m:{type}{{{type}: "{record_uuid}"}}) MERGE (n)-[r:HAS_RECORD]->(m)'''
    print(add_str)
    return add_str
