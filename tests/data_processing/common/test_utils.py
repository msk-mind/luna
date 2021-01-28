from data_processing.common.utils import *

def test_generate_uuid():
    uuid = generate_uuid("file:./tests/data_processing/common/test_config.yml", ["FEATURE", "label"])

    assert uuid.startswith("FEATURE-label-")
