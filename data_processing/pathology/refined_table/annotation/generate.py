
import click

from data_processing.common.CodeTimer import CodeTimer
from data_processing.common.config import ConfigSet
from data_processing.common.custom_logger import init_logger
from data_processing.common.sparksession import SparkConfig
from data_processing.common.Neo4jConnection import Neo4jConnection
from data_processing.common.utils import generate_uuid_dict
import data_processing.common.constants as const
from data_processing.pathology.common.build_geojson_from_bitmap import build_geojson_from_bitmap, concatenate_regional_geojsons
from data_processing.pathology.common.utils import get_add_triple_str

from pyspark.sql.functions import udf, lit, col, first, last, desc, array, to_json, collect_list, current_timestamp
from pyspark.sql.window import Window
from pyspark.sql.types import StringType, IntegerType, ArrayType, MapType, StructType, StructField

import yaml, os, json

logger = init_logger()

geojson_struct = StructType([
    StructField("type", StringType()),
    StructField("features",
                ArrayType(
                    StructType([
                        StructField("type", StringType()),
                        StructField("properties", MapType(StringType(), StringType())),
                        StructField("geometry",
                                    StructType([
                                        StructField("type", StringType()),
                                        StructField("coordinates", ArrayType(ArrayType(IntegerType())))
                                    ])
                                    )
                    ])
                )
                )
])

@click.command()
@click.option('-t', '--template_file', default=None, type=click.Path(exists=True),
              help="path to yaml template file containing information required for pathology proxy data ingestion. "
                   "See data_ingestion_template.yaml.template")
@click.option('-f', '--config_file', default='config.yaml', type=click.Path(exists=True),
              help="path to config file containing application configuration. See config.yaml.template")
@click.option('-l', '--label_file', default='data_processing/pathology/config/regional_etl_configuration.yaml',
              type=click.Path(exists=True),
              help="path to label configuration file containing application configuration.")
@click.option('-p', '--process_string', default='geojson',
              help='comma separated list of processes to run or replay: e.g. geojson OR concat')
def cli(template_file, config_file, label_file, process_string):
    """
    This module generates geojson table for pathology data based on information specified in the template file.

    Example:
        python -m data_processing.pathology.refined_table.annotation.generate \
        --template_file {PATH_TO_TEMPLATE_FILE} \
        --config_file {PATH_TO_CONFIG_FILE} \
        --label-file {PATH_TO_LABEL_FILE} \
        --process_string geojson

    """
    with CodeTimer(logger, 'generate GEOJSON table'):
        processes = process_string.lower().strip().split(",")
        logger.info('data template: ' + template_file)
        logger.info('config_file: ' + config_file)
        logger.info('processes: ' + str(processes))

        # load configs
        cfg = ConfigSet(name=const.DATA_CFG, config_file=template_file)
        cfg = ConfigSet(name=const.APP_CFG,  config_file=config_file)

        if 'geojson' in processes:
            exit_code = create_geojson_table(label_file)
            if exit_code != 0:
                logger.error("GEOJSON table creation had errors. Exiting.")
                return

        if 'concat' in processes:
            exit_code = create_concat_geojson_table()
            if exit_code != 0:
                logger.error("CONCAT-GEOJSON table creation had errors. Exiting.")
                return

def create_geojson_table(label_file):
    """
    Vectorizes npy array annotation file into polygons and builds GeoJson with the polygon features.
    Creates a geojson file per labelset.
    """
    exit_code = 0

    cfg = ConfigSet()
    spark = SparkConfig().spark_session(config_name=const.APP_CFG, app_name="data_processing.pathology.refined_table.annotation.generate")
    conn = Neo4jConnection(uri=cfg.get_value(path=const.DATA_CFG+'::GRAPH_URI'),
                           user=cfg.get_value(path=const.DATA_CFG+'::GRAPH_USER'),
                           pwd=cfg.get_value(path=const.DATA_CFG+'::GRAPH_PW'))

    # load paths from configs
    bitmask_table_path = const.TABLE_LOCATION(cfg, is_source=True)
    geojson_table_path = const.TABLE_LOCATION(cfg)

    df = spark.read.format("delta").load(bitmask_table_path)

    # need bitmask df if numpy generation not run, read in bitmask table currently available
    df = df.toPandas()
    logger.info(df)

    # explode table by labelsets
    with open(label_file) as labelfile:
        label_config = yaml.safe_load(labelfile)

    label_config = label_config[cfg.get_value(path=const.DATA_CFG+'::PROJECT')]
    labelsets = [cfg.get_value(path=const.DATA_CFG+'::USE_LABELSET')]

    if cfg.get_value(path=const.DATA_CFG+'::USE_ALL_LABELSETS'):
        labelsets = list(label_config['label_sets'].keys())

    df["labelset"] = [labelsets] * len(df)
    df = df.explode('labelset')

    df = spark.createDataFrame(df)

    # setup variables needed for build geojson UDF
    contour_level = cfg.get_value(path=const.DATA_CFG+'::CONTOUR_LEVEL')
    polygon_tolerance = cfg.get_value(path=const.DATA_CFG+'::POLYGON_TOLERANCE')
    dmt = cfg.get_value(path=const.DATA_CFG+'::PROJECT')
    configuration_file = os.path.join(os.getcwd(), label_file)

    # populate geojson and geojson_record_uuid
    build_geojson_from_bitmap_udf = udf(build_geojson_from_bitmap, geojson_struct)
    geojson_record_uuid_udf = udf(generate_uuid_dict, StringType())
    spark.sparkContext.addPyFile("./data_processing/common/EnsureByteContext.py")

    # cache to not have udf called multiple times
    df = df.withColumn("geojson",
                       build_geojson_from_bitmap_udf(lit(configuration_file),lit(dmt),"npy_filepath","labelset",lit(contour_level),lit(polygon_tolerance))) \
            .cache()
    df = df.withColumn("geojson_record_uuid", geojson_record_uuid_udf(to_json("geojson"), array(lit("SVGEOJSON"), "labelset")))

    # drop empty geojsons that may have been created
    df = df.dropna()

    # build refined table by selecting columns from output table
    geojson_df = df.select("sv_project_id", "slideviewer_path", "slide_id", "bmp_record_uuid", "user", "labelset", "geojson", "geojson_record_uuid")
    geojson_df = geojson_df.withColumn("latest", lit(True))        \
                         .withColumn("date_added", current_timestamp())    \
                         .withColumn("date_updated", current_timestamp())
    # create geojson delta table
    # update main table if exists, otherwise create new table
    if not os.path.exists(geojson_table_path):
        geojson_df.write.format("delta").save(geojson_table_path)
    else:
        from delta.tables import DeltaTable
        existing_geojson_table = DeltaTable.forPath(spark, geojson_table_path)
        existing_geojson_table.alias("main_geojson_table") \
            .merge(geojson_df.alias("geojson_annotation_updates"), "main_geojson_table.geojson_record_uuid = geojson_annotation_updates.geojson_record_uuid") \
            .whenMatchedUpdate(set = { "date_updated" : "geojson_annotation_updates.date_updated" } ) \
            .whenNotMatchedInsertAll() \
            .execute()

        # Removed synclatest flag - always update when merging with existing table.
        # add latest flag
        windowSpec = Window.partitionBy("user", "slide_id", "labelset").orderBy(desc("date_updated"))
        # Note that last != opposite of first! Have to use desc ordering with first...
        spark.read.format("delta").load(geojson_table_path) \
            .withColumn("date_latest", first("date_updated").over(windowSpec)) \
            .withColumn("latest", col("date_latest")==col("date_updated")) \
            .drop("date_latest") \
            .write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(geojson_table_path)

    logger.info("Finished building Geojson table.")

    # Add relationships to graph
    geojson_table = spark.read.format("delta").load(geojson_table_path).select("slide_id","geojson_record_uuid").toPandas()
    geojson_table.apply(lambda x: conn.query(get_add_triple_str(x.slide_id, "geojson_record_uuid", x.geojson_record_uuid)), axis=1)
    logger.info("Updated graph with new geojson records.")

    return exit_code


def create_concat_geojson_table():
    """
    Aggregate geojson features for each labelset, in case there are annotations from multiple users.
    """
    exit_code = 0

    cfg = ConfigSet()
    spark = SparkConfig().spark_session(config_name=const.APP_CFG, app_name="data_processing.pathology.refined_table.annotation.generate")
    spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "false")
    conn = Neo4jConnection(uri=cfg.get_value(path=const.DATA_CFG+'::GRAPH_URI'),
                           user=cfg.get_value(path=const.DATA_CFG+'::GRAPH_USER'),
                           pwd=cfg.get_value(path=const.DATA_CFG+'::GRAPH_PW'))

    # load paths from config
    geojson_table_path = const.TABLE_LOCATION(cfg, is_source=True)
    concat_geojson_table_path = const.TABLE_LOCATION(cfg)

    concatgeojson_df = spark.read.format("delta").load(geojson_table_path)
    concatgeojson_df = concatgeojson_df.toPandas()

    # only use latest annotations for concatenating.
    concatgeojson_df = concatgeojson_df.loc[(concatgeojson_df['latest'] == True)]

    concatgeojson_df = spark.createDataFrame(concatgeojson_df)

    # make geojson string list for slide + labelset
    concatgeojson_df = concatgeojson_df \
        .select("sv_project_id", "slideviewer_path", "slide_id", "labelset", to_json("geojson").alias("geojson_str")) \
        .groupby(["sv_project_id", "slideviewer_path", "slide_id", "labelset"]) \
        .agg(collect_list("geojson_str").alias("geojson_list"))

    # set up udfs
    concatenate_regional_geojsons_udf = udf(concatenate_regional_geojsons, geojson_struct)
    concat_geojson_record_uuid_udf = udf(generate_uuid_dict, StringType())
    spark.sparkContext.addPyFile("./data_processing/common/EnsureByteContext.py")

    concatgeojson_df = concatgeojson_df.withColumn("concat_geojson", concatenate_regional_geojsons_udf("geojson_list")).cache()

    concatgeojson_df = concatgeojson_df \
        .drop("geojson_list") \
        .withColumn("concat_geojson_record_uuid", concat_geojson_record_uuid_udf(to_json("concat_geojson"), array(lit("SVCONCATGEOJSON"), "labelset")))
    
    concatgeojson_df = concatgeojson_df.withColumn("latest", lit(True))   \
                            .withColumn("date_added", current_timestamp())    \
                            .withColumn("date_updated", current_timestamp())

    # create concatenation geojson delta table
    # update main table if exists, otherwise create main table
    if not os.path.exists(concat_geojson_table_path):
        concatgeojson_df.write.format("delta").save(concat_geojson_table_path)
    else:
        from delta.tables import DeltaTable
        concat_geojson_table = DeltaTable.forPath(spark, concat_geojson_table_path)
        concat_geojson_table.alias("main_geojson_table") \
            .merge(concatgeojson_df.alias("geojson_annotation_updates"), "main_geojson_table.concat_geojson_record_uuid = geojson_annotation_updates.concat_geojson_record_uuid") \
            .whenMatchedUpdate(set = { "date_updated" : "geojson_annotation_updates.date_updated" } ) \
            .whenNotMatchedInsertAll() \
            .execute()

        # add latest flag
        windowSpec = Window.partitionBy("user", "slide_id", "labelset").orderBy(desc("date_updated"))
        # Note that last != opposite of first! Have to use desc ordering with first...
        spark.read.format("delta").load(concat_geojson_table_path) \
            .withColumn("date_latest", first("date_updated").over(windowSpec)) \
            .withColumn("latest", col("date_latest")==col("date_updated")) \
            .drop("date_latest") \
            .write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(concat_geojson_table_path)

    logger.info("Finished building Concatenation table.")

    # Add relationships to graph
    concat_geojson_table = spark.read.format("delta").load(concat_geojson_table_path).select("slide_id","concat_geojson_record_uuid").toPandas()
    concat_geojson_table.apply(lambda x: conn.query(get_add_triple_str(x.slide_id, "concat_geojson_record_uuid", x.concat_geojson_record_uuid)), axis=1)
    logger.info("Updated graph with new concat geojson records.")

    return exit_code


if __name__ == "__main__":
    cli()
