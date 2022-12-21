import click
import time
from pyspark.sql import functions as F

from luna.common.GraphEnum import GraphEnum
from luna.common.Node import Node
from luna.common.utils import clean_nested_colname
from luna.common.Neo4jConnection import Neo4jConnection
from luna.common.custom_logger import init_logger
from luna.common.sparksession import SparkConfig
from luna.common.config import ConfigSet
import luna.common.constants as const


@click.command()
@click.option(
    "-d",
    "--data_config_file",
    default="luna/services/config.yaml",
    required=True,
    help="path to data configuration related to package. See data_config.yaml.template in this package.",
)
@click.option(
    "-a",
    "--app_config_file",
    default="config.yaml",
    required=True,
    help="path to config file containing application configuration. See config.yaml.template",
)
def update_graph(data_config_file, app_config_file):
    """
    Updates graph with the data from the table,
    Usage:
            python3 -m luna.services.graph_service -d data-config.yaml -f config.yaml
    """
    logger = init_logger("update_graph.log")
    start_time = time.time()

    # Set up : Neo4j connection and Spark session
    logger.info("Setting up graph connection and spark session")
    cfg = ConfigSet(name=const.APP_CFG, config_file=app_config_file)
    cfg = ConfigSet(name=const.DATA_CFG, config_file=data_config_file)
    spark = SparkConfig().spark_session(
        config_name=const.APP_CFG, app_name="update-graph"
    )

    conn = Neo4jConnection(
        uri=cfg.get_value(path=const.DATA_CFG + "::GRAPH_URI"),
        user=cfg.get_value(path=const.DATA_CFG + "::GRAPH_USER"),
        pwd=cfg.get_value(path=const.DATA_CFG + "::GRAPH_PW"),
    )

    # get project / table path
    PROJECT = cfg.get_value(path=const.DATA_CFG + "::PROJECT")

    # load table
    DATA_TYPE = cfg.get_value(path=const.DATA_CFG + "::DATA_TYPE")
    table_path = const.TABLE_LOCATION(cfg)
    df = spark.read.format("delta").load(table_path)

    # get graph info
    data_type = DATA_TYPE.upper()
    graphs = GraphEnum[data_type].value
    # graph ~= relationship
    for graph in graphs:

        src_node_type = graph.src.type
        src_node_fields = graph.src.get_all_schema()

        relationship = graph.relationship

        target_node_type = graph.target.type
        target_node_fields = graph.target.get_all_schema()

        logger.info(
            "Update graph with {0} - {1} - {2}".format(
                src_node_type, relationship, target_node_type
            )
        )

        # subset dataframe
        src_alias = [(field, clean_nested_colname(field)) for field in src_node_fields]
        target_alias = [
            (field, clean_nested_colname(field)) for field in target_node_fields
        ]
        fields_alias = list(set(src_alias + target_alias))

        pdf = (
            df.select([F.col(c).alias(a) for c, a in fields_alias])
            .groupBy([alias for field, alias in fields_alias])
            .count()
            .toPandas()
        )

        count = 0
        # update graph
        for index, row in pdf.iterrows():

            src_props = {}
            for _, sa in src_alias:
                src_props[sa] = row[sa]
            src_props["namespace"] = PROJECT
            src_node = Node(
                src_node_type,
                src_props.pop(clean_nested_colname(graph.src.name)),
                src_props,
            )

            target_props = {}
            for _, ta in target_alias:
                target_props[ta] = row[ta]
            target_props["namespace"] = PROJECT
            target_node = Node(
                target_node_type,
                target_props.pop(clean_nested_colname(graph.target.name)),
                target_props,
            )

            try:
                query = f"""MERGE (n:{src_node.get_create_str()}) MERGE (m:{target_node.get_create_str()}) MERGE (n)-[r:{relationship}]->(m)"""
                conn.query(query)
            except Exception as ex:
                query = f"""MATCH (n:{src_node.get_match_str()}) MERGE (m:{target_node.get_match_str()}) MERGE (n)-[r:{relationship}]->(m)"""
                conn.query(query)
            logger.info(query)
            count = index
        logger.info(f"Updated {count+1} nodes")
    logger.info("Finished update-graph in %s seconds" % (time.time() - start_time))


if __name__ == "__main__":
    update_graph()
