from luna.common.utils import generate_uuid, get_absolute_path


def test_generate_uuid():
    uuid = generate_uuid("./tests/luna/common/test_config.yml", ["FEATURE", "label"])

    assert uuid.startswith("FEATURE-label-")


def test_get_absolute_path():
    absolute_path = get_absolute_path(
        __file__, "../data_ingestion_template_invalid.yml"
    )
    assert absolute_path.startswith("/")
    assert absolute_path.endswith("tests/luna/data_ingestion_template_invalid.yml")
