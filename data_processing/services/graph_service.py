import os
import click
import time
import pandas as pd
from pyspark.sql import functions as F

from data_processing.common.GraphEnum import GraphEnum
from data_processing.common.Neo4jConnection import Neo4jConnection
from data_processing.common.custom_logger import init_logger
from data_processing.common.sparksession import SparkConfig
from data_processing.common.config import ConfigSet
import data_processing.common.constants as const


def prop_str(fields, row):
	"""
	Returns a kv string like 'id: 123, ...' where prop values come from row.
	"""
	kv = [f" {x}: '{row[x]}'" for x in fields]
	return ','.join(kv)

@click.command()
@click.option('-d', '--data_config_file', default = 'data_processing/services/config.yaml', required=True,
		help="path to configuration related to package. See config.yaml.template in this package.")
@click.option('-f', '--app_config_file', default = 'config.yaml', required=True,
              help="path to config file containing application configuration. See config.yaml.template")
def update_graph(data_config_file, app_config_file):
	"""
	Updates graph with the data from the table, 
	Usage:
		python3 -m data_processing.services.graph_service -p TEST_PROJECT -t dicom -f config.yaml
	"""
	logger = init_logger('update_graph.log')
	start_time = time.time()

	# Set up : Neo4j connection and Spark session
	logger.info("Setting up graph connection and spark session")
	cfg = ConfigSet(name=const.APP_CFG, config_file=app_config_file)
	cfg = ConfigSet(name=const.DATA_CFG, config_file=data_config_file)
	spark = SparkConfig().spark_session(config_name=const.APP_CFG, app_name="update-graph")

	conn = Neo4jConnection(uri=cfg.get_value(path=const.DATA_CFG+'::GRAPH_URI'),
		user=cfg.get_value(path=const.DATA_CFG+'::GRAPH_USER'),
		pwd=cfg.get_value(path=const.DATA_CFG+'::GRAPH_PW'))

	# get project / table path
	base_path = cfg.get_value(path=const.DATA_CFG+'::MIND_DATA_PATH')
	project_dir = os.path.join(base_path, cfg.get_value(path=const.DATA_CFG+'::PROJECT_NAME'))
	logger.info("Got project path : " + project_dir)

	# load table
	TABLE_NAME = cfg.get_value(path=const.DATA_CFG+'::TABLE_NAME')
	table_path = os.path.join(project_dir, const.TABLE_DIR, TABLE_NAME)
	df = spark.read.format("delta").load(table_path)

	# get graph info
	table = TABLE_NAME.upper()
	graphs = GraphEnum[table].value
	# graph ~= relationship
	for graph in graphs:

		src_node_type = graph.src.type
		src_node_fields = graph.src.schema

		relationship = graph.relationship

		target_node_type = graph.target.type
		target_node_fields = graph.target.schema

		logger.info("Update graph with {0} - {1} - {2}".format(src_node_type, relationship, target_node_type))

		# subset dataframe
		fields = src_node_fields + target_node_fields
		# alias removes top-level column name if '.' is part of the column name: i.e. metadata.PatientName -> PatientName
		src_alias = [x[x.find('.')+1:] for x in src_node_fields]
		target_alias = [x[x.find('.')+1:] for x in target_node_fields]
		fields_alias = src_alias + target_alias

		pdf = df.select([F.col(c).alias(a) for c, a in zip(fields, fields_alias)]) \
			.groupBy(fields_alias) \
			.count() \
			.toPandas()

		# update graph
		for index, row in pdf.iterrows():

			src_props = prop_str(src_alias, row)
			target_props = prop_str(target_alias, row)

			# fire query! # match on "ID" in case of update?
			query = f'''MERGE (n:{src_node_type} {{{src_props}}}) MERGE (m:{target_node_type} {{{target_props}}}) MERGE (n)-[r:{relationship}]->(m)'''
			logger.info(query)
			conn.query(query)
			
	logger.info("Finished update-graph in %s seconds" % (time.time() - start_time))

if __name__ == "__main__":
	update_graph()
