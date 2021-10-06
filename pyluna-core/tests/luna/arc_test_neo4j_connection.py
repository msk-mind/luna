from neo4j import GraphDatabase, Record
from neo4j import __version__ as neo4j_version
import pytest
from pytest_mock import mocker

from luna.common.Neo4jConnection import Neo4jConnection, pretty_path
from luna.common.config import ConfigSet
from luna.common.sparksession import SparkConfig
from pyspark import SQLContext, SparkContext
'''
Mock tests for Neo4jConnection in common
'''

@pytest.fixture(autouse=True)
def spark():
    print('------setup------')
    APP_CFG = 'APP_CFG'
    ConfigSet(name=APP_CFG, config_file='tests/test_config.yml')
    spark = SparkConfig().spark_session(config_name=APP_CFG, app_name='test-neo4j-connection')

    yield spark

    print('------teardown------')


def test_create_id_lookup_table_where(mocker, spark):

	sqlc = SQLContext(spark)

	# mock data
	mocker.patch.object(Neo4jConnection, 'query')
	record1 = {'source': {'value': 'P-123'}, 'sink': {'value': '1.1.1'}, 'path': [{'value': 'P-123'}, 'ID_LINK', {'value': 'RIA_11-111_111'}, 'ID_LINK', {'value': '1.1.1'}]}
	record2 = {'source': {'value': 'P-123'}, 'sink': {'value': '1.2.2'}, 'path': [{'value': 'P-123'}, 'ID_LINK', {'value': 'RIA_11-111_222'}, 'ID_LINK', {'value': '1.2.2'}]}
	r1 = Record(record1)
	r2 = Record(record2)
	query_result = [r1, r2]
	Neo4jConnection.query.return_value = query_result

	mocker.patch.object(Neo4jConnection, 'create_id_lookup_table_where')
	Neo4jConnection.create_id_lookup_table_where.return_value = sqlc.createDataFrame([('P-123', '1.1.1', 'some_path1'), ('P-123', '1.2.2', 'some_path2')], 
		['dmp_patient_id', 'SeriesInstanceUID', 'pathspec'])

	df = Neo4jConnection.create_id_lookup_table_where(sqlc, 
		source='dmp_patient_id',  
		sink='SeriesInstanceUID',  
		WHERE_CLAUSE="WHERE source.value='P-0037794'")

	assert 2 == df.count()
	assert set(['dmp_patient_id', 'SeriesInstanceUID', 'pathspec']) == set(df.columns)


def test_create_id_lookup_table_where_banned_words(mocker, spark):

	sqlc = SQLContext(spark)

	mocker.patch.object(Neo4jConnection, 'create_id_lookup_table_where', side_effect="You tried to alter the database, goodbye")

	with pytest.raises(Exception) as exe:
		Neo4jConnection.create_id_lookup_table_where(sqlc, 
		source='dmp_patient_id',  
		sink='SeriesInstanceUID',  
		WHERE_CLAUSE="DELETE (n:test)")

		assert "You tried to alter the database, goodbye" == exec.value.message


def test_pretty_print():

	path = [{'value': 'P-123'}, 'ID_LINK', {'value': 'RIA_11-111_111'}, 'ID_LINK', {'value': '1.1.1'}]
	
	pretty_path_result = pretty_path(path)

	assert "SOURCE:(value:P-123)-[ID_LINK]-(value:RIA_11-111_111)-[ID_LINK]-SINK:(value:1.1.1)" == pretty_path_result


