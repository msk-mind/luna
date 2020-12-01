import os
import click
import pandas as pd
from pyspark.sql import functions as F

from data_processing.common.GraphEnum import GraphEnum
from data_processing.common.Neo4jConnection import Neo4jConnection
from data_processing.common.custom_logger import init_logger
from data_processing.common.sparksession import SparkConfig
from data_processing.common.config import ConfigSet
import data_processing.common.constants as const

@click.command()
@click.option('-p', '--project', help="project name", required=True)
@click.option('-t', '--table', help="table name", required=True)
@click.option('-f', '--config_file', default = 'config.yaml', required=True,
              help="path to config file containing application configuration. See config.yaml.template")
def update_graph(project, table, config_file):

	logger = init_logger('update_graph.log')

	# Set up : Neo4j connection and Spark session
	logger.info("Setting up graph connection and spark session")
	cfg = ConfigSet(name=const.APP_CFG, config_file=config_file)
	spark = SparkConfig().spark_session(config_name=const.APP_CFG, app_name="update-graph")

	conn = Neo4jConnection(uri=cfg.get_value(name=const.APP_CFG, jsonpath='GRAPH_URI'),
		user=cfg.get_value(name=const.APP_CFG, jsonpath='GRAPH_USER'),
		pwd=cfg.get_value(name=const.APP_CFG, jsonpath='GRAPH_PW'))

	# get project / table path
	if "MIND_ROOT_DIR" in os.environ:
		base_path = os.environ["MIND_ROOT_DIR"]
	else:
		base_path = "/gpfs/mskmindhdp_emc/data/"
	project_dir = os.path.join(base_path, project)
	logger.info("Got project path : " + project_dir)

	# load table
	table_path = os.path.join(project_dir, const.TABLE_DIR, table)
	df = spark.read.format("delta").load(table_path)

	# get graph info
	table = table.upper()
	graphs = GraphEnum[table].value

	for graph in graphs:
		src = graph.src
		relationship = graph.relationship
		target = graph.target
		src_column_name = graph.src_column_name
		target_column_name = graph.target_column_name

		# update graph
		logger.info("Update graph with {0} - {1} - {2}".format(src, relationship, target))

		def add_to_graph(record: pd.DataFrame) -> pd.DataFrame:
			conn.query(f'''MERGE (n:"{src}"{{"{src}": "{record[src]}"}}) MERGE (m:"{target}"{{"{target}": "{record[target]}"}}) MERGE (n)-[r:"{relationship}"]->(m)''')

		# subset dataframe
		df = df.select(F.col(src_column_name).alias(src), F.col(target_column_name).alias(target))

		df.groupBy(src).applyInPandas(add_to_graph, schema=df.schema)

if __name__ == "__main__":
	update_graph()