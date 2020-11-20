
from data_processing.common import GraphEnum, Neo4jConnection

def update_graph(project_name, table_name):

	# Neo4j connection

	# get project dir path

	# load table

	# get graph info
	table_name = table_name.upper()
	graph = GraphEnum[table_name]

	# filter table with Graph.src_column_name, target_column_name etc
	# and populate graph query

	# fire the query!