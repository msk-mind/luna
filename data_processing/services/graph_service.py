from data_processing.common import GraphEnum, Neo4jConnection, custom_logger
from data_processing.common.sparksession import SparkConfig
import data_processing.common.constants as const

# TODO test with test-project.
@click.command()
@click.option('-p', '--project', help="project name", required=True)
@click.option('-t', '--table', help="table name", required=True)
@click.option('-f', '--config_file', default = 'config.yaml', required=True,
              help="path to config file containing application configuration. See config.yaml.template")
def update_graph(project_name, table_name, config_file):

	logger = init_logger('update_graph.log')

	# Set up : Neo4j connection and Spark session
	logger.info("Setting up graph connection and spark session")
	GRAPH_URI = os.environ["GRAPH_URI"]
	conn = Neo4jConnection(uri=GRAPH_URI, user="neo4j", pwd="password")

	spark = SparkConfig().spark_session(config_name=config_file, app_name="update-graph")

	# get project / table path
	if "BASE_PATH" in os.environ:
		base_path = os.environ["BASE_PATH"]
	else:
		base_path = "/gpfs/mskmindhdp_emc/data/"
	project_dir = os.path.join(base_path, project_name)
	logger.info("Got project path : " + project_dir)

	# load table
	table_path = os.path.join(project_dir, const.BASE, table_name)
	df = spark.read.format("delta").load(table_path)

	# get graph info
	table_name = table_name.upper()
	graphs = GraphEnum[table_name].value

	for graph in graphs:
		src = graph.src
		relationship = graph.relationship
		target = graph.target
		src_column_name = graph.src_column_name
		target_column_name = graph.target_column_name

		# update graph
		logger.info("Update graph with {0} - {1} - {2}".format(src, relationship, target))

		def add_to_graph(record: pd.DataFrame) -> pd.DataFrame:
		    conn.query(f'''MERGE (n:"{src}"{{"{src}": "{record[src_column_name]}"}}) MERGE (m:"{target}"{{"{target}": "{record[target_column_name]}"}}) MERGE (n)-[r:"{relationship}"]->(m)''')

		df = df.groupBy(src_column_name).applyInPandas(add_to_graph, schema=df.schema)

if __name__ == "__main__":
	update_graph()