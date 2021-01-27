from data_processing.pathology.common.utils import *

def test_get_add_triple_str():
    res = get_add_triple_str("123", "geojson_record_uuid", "SVGEOJSON-default-123")

    assert 'MERGE (n:slide{slide_id: "123"}) MERGE (m:geojson_record_uuid{geojson_record_uuid: "SVGEOJSON-default-123"}) MERGE (n)-[r:HAS_RECORD]->(m)' == res
