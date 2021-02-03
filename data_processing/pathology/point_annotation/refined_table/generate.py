"""
The steps for processing pathology nuclei point annotations includes:
1) (proxy_table) JSON Anotations are first downloaded from slideviewer
2) (refined_table) JSON annotations are then converted into qupath-compatible geojson files
"""

import os, json
import click
from pyspark.sql.window import Window
from pyspark.sql.functions import first, last, col, lit, desc, udf, explode, array, to_json, current_timestamp
from pyspark.sql.types import ArrayType, StringType, MapType, IntegerType, StructType, StructField

from data_processing.common.CodeTimer import CodeTimer
from data_processing.common.config import ConfigSet
from data_processing.common.custom_logger import init_logger
from data_processing.common.sparksession import SparkConfig
from data_processing.common.utils import generate_uuid_dict
from data_processing.common.build_geojson 	import build_geojson_from_pointclick_json
import data_processing.common.constants as const

os.environ['OPENBLAS_NUM_THREADS'] = '1'

logger = init_logger()

# geojson struct - example
# [{"type": "Feature", "id": "PathAnnotationObject", "geometry": {"type": "Point", "coordinates": [13981, 15274]}, "properties": {"classification": {"name": "Other"}}},
# {"type": "Feature", "id": "PathAnnotationObject", "geometry": {"type": "Point", "coordinates": [14013, 15279]}, "properties": {"classification": {"name": "Other"}}}]
geojson_struct = ArrayType(
    StructType([
        StructField("type", StringType()),
        StructField("id", StringType()),
        StructField("geometry",
                    StructType([
                        StructField("type", StringType()),
                        StructField("coordinates", ArrayType(IntegerType()))
                    ])
        ),
        StructField("properties", MapType(StringType(), MapType(StringType(), StringType())))
    ])
)


# todo: move to common module
def get_labelset_mapping():
    cfg = ConfigSet()
    label_config = cfg.get_value(path=const.DATA_CFG+'::LABEL_SETS')
    labelset_mapping = label_config[cfg.get_value(path=const.DATA_CFG+'::USE_LABELSET')]
    labelset_mapping_ret = labelset_mapping.copy()
    return labelset_mapping_ret

def get_labelset_keys():
    cfg = ConfigSet()
    label_config = cfg.get_value(path=const.DATA_CFG+'::LABEL_SETS')
    labelsets = [cfg.get_value(path=const.DATA_CFG+'::USE_LABELSET')]

    if cfg.get_value(path=const.DATA_CFG+'::USE_ALL_LABELSETS'):
        labelsets = list(label_config.keys())

    return labelsets


@click.command()
@click.option('-t', '--data_config_file', default=None, type=click.Path(exists=True),
              help="path to data yaml file containing information required to build pathology point annotation geojsons. "
                   "See data-config.yaml.template")
@click.option('-a', '--app_config_file', default='config.yaml', type=click.Path(exists=True),
              help="path to config file containing application configuration. See config.yaml.template")
def cli(data_config_file, app_config_file):
    """
    This module generates geojson table for pathology data based on information specified in the template file.

    Example:
        python -m data_processing.pathology.point_annotation.refined_table.generate \
        --data_config_file {PATH_TO_DATA_CONFIG_FILE} \
        --app_config_file {PATH_TO_APP_CONFIG_FILE}

    """
    with CodeTimer(logger, 'generate GEOJSON table'):
        logger.info('data config file: ' + data_config_file)
        logger.info('app config file: ' + app_config_file)

        # load configs
        cfg = ConfigSet(name=const.DATA_CFG, config_file=data_config_file)
        cfg = ConfigSet(name=const.APP_CFG,  config_file=app_config_file)

        create_refined_table()


def create_refined_table():

    cfg = ConfigSet()
    spark = SparkConfig().spark_session(config_name=const.APP_CFG, app_name="data_processing.pathology.point_annotation.proxy_table.generate")

    # load paths from configs
    point_table_path = const.TABLE_LOCATION(cfg, is_source=True)
    geojson_table_path = const.TABLE_LOCATION(cfg)

    df = spark.read.format("delta").load(point_table_path)

    labelsets = get_labelset_keys()
    labelset_names = array([lit(key) for key in labelsets])

    # explode labelsets
    df = df.withColumn("labelset_list", labelset_names) \
            .select("slideviewer_path", "slide_id", "sv_project_id", "sv_json_record_uuid", "user", "sv_json", explode("labelset_list").alias("labelset"))
    df.show()

    # build geojsons
    label_config = cfg.get_value(path=const.DATA_CFG+'::LABEL_SETS')

    build_geojson_from_pointclick_json_udf = udf(build_geojson_from_pointclick_json,  geojson_struct)
    df = df.withColumn("geojson", build_geojson_from_pointclick_json_udf(lit(str(label_config)), "labelset", "sv_json")).cache()

    # populate "date_added", "date_updated","latest", "sv_json_record_uuid"
    geojson_record_uuid_udf = udf(generate_uuid_dict, StringType())
    df = df.withColumn("geojson_record_uuid", geojson_record_uuid_udf(to_json("geojson"), array(lit("SVPTGEOJSON"), "labelset"))) \
        .withColumn("latest", lit(True)) \
        .withColumn("date_added", current_timestamp()) \
        .withColumn("date_updated", current_timestamp())

    # create geojson delta table
    # update main table if exists, otherwise create main table
    if not os.path.exists(geojson_table_path):
        df.write.format("delta").save(geojson_table_path)
    else:
        from delta.tables import DeltaTable
        geojson_table = DeltaTable.forPath(spark, geojson_table_path)
        geojson_table.alias("main_pt_geojson_table") \
            .merge(df.alias("pt_geojson_annotation_updates"), "main_pt_geojson_table.geojson_record_uuid = pt_geojson_annotation_updates.geojson_record_uuid") \
            .whenMatchedUpdate(set = { "main_pt_geojson_table.date_updated" : "pt_geojson_annotation_updates.date_updated" } ) \
            .whenNotMatchedInsertAll() \
            .execute()

    # add latest flag
    windowSpec = Window.partitionBy("user", "slide_id", "labelset").orderBy(desc("date_updated"))
    # Note that last != opposite of first! Have to use desc ordering with first...
    spark.read.format("delta").load(geojson_table_path) \
        .withColumn("date_latest", first("date_updated").over(windowSpec)) \
        .withColumn("latest", col("date_latest")==col("date_updated")) \
        .drop("date_latest") \
        .write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(geojson_table_path)
