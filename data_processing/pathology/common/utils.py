
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
