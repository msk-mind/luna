import shutil

import click

from data_processing.common.CodeTimer import CodeTimer
from data_processing.common.config import ConfigSet
from data_processing.common.custom_logger import init_logger
from data_processing.common.sparksession import SparkConfig
import data_processing.common.constants as const
from data_processing.pathology.common.utils import get_labelset_keys

from pyspark.sql.functions import udf, lit, col, first, last, desc, array, to_json, collect_list, current_timestamp, explode
from pyspark.sql.window import Window
from pyspark.sql.types import StringType, IntegerType, ArrayType, MapType, StructType, StructField

import yaml, os, json

os.environ['OPENBLAS_NUM_THREADS'] = '1'
logger = init_logger()

# geojson struct - example
# {"type":"FeatureCollection",
# "features":[{"type":"Feature",
#               "properties":{"label_num":"1","label_name":"tissue_1"},
#               "geometry":{"type":"Polygon",
#                           "coordinates":[
#                                       [[1261,2140],[1236,2140],[1222,2134],[1222,2132],[1216,2125]],
#                                       [[hole1_x,hole_y], [hole2_x, hole3_x], ...]
#                                       ]
#                           }
#             }]
# }
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
                                        StructField("coordinates", ArrayType(ArrayType(ArrayType(IntegerType()))))
                                    ])
                                    )
                    ])
                )
                )
])

# Base template for geoJSON file
geojson_base = {
    "type": "FeatureCollection",
    "features": []
}

# max amount of time for a geojson to be generated. if generation surpasses this limit, it is likely the annotation file is
# too large or they may be annotation artifacts present in the slide. currently set at 30 minute timeout
TIMEOUT_SECONDS = 1800


@click.command()
@click.option('-d', '--data_config_file', default=None, type=click.Path(exists=True),
              help="path to yaml file containing data input and output parameters. "
                   "See ./data_config.yaml.template")
@click.option('-a', '--app_config_file', default='config.yaml', type=click.Path(exists=True),
              help="path to yaml file containing application runtime parameters. "
                   "See ./app_config.yaml.template")
@click.option('-p', '--process_string', default='geojson',
              help='process to run or replay: e.g. geojson OR concat')
def cli(data_config_file, app_config_file, process_string):
    """
        This module generates a delta table with geojson pathology data based on the input and output parameters
         specified in the data_config_file.

        Example:
            python3 -m data_processing.pathology.refined_table.regional_annotation.generate \
                     --data_config_file <path to data config file> \
                     --app_config_file <path to app config file> \
                     --process_string geojson
    """
    with CodeTimer(logger, f"generate {process_string} table"):
        logger.info('data template: ' + data_config_file)
        logger.info('config_file: ' + app_config_file)

        # load configs
        cfg = ConfigSet(name=const.DATA_CFG, config_file=data_config_file)
        cfg = ConfigSet(name=const.APP_CFG,  config_file=app_config_file)

        # copy app and data configuration to destination config dir
        config_location = const.CONFIG_LOCATION(cfg)
        os.makedirs(config_location, exist_ok=True)

        shutil.copy(app_config_file, os.path.join(config_location, "app_config.yaml"))
        shutil.copy(data_config_file, os.path.join(config_location, "data_config.yaml"))
        logger.info("config files copied to %s", config_location)

        if 'geojson' == process_string:
            exit_code = create_geojson_table()
            if exit_code != 0:
                logger.error("GEOJSON table creation had errors. Exiting.")
                return

        if 'concat' == process_string:
            exit_code = create_concat_geojson_table()
            if exit_code != 0:
                logger.error("CONCAT-GEOJSON table creation had errors. Exiting.")
                return

def create_geojson_table():
    """
    Vectorizes npy array annotation file into polygons and builds GeoJson with the polygon features.
    Creates a geojson file per labelset.
    """
    exit_code = 0

    cfg = ConfigSet()
    spark = SparkConfig().spark_session(config_name=const.APP_CFG,
                                        app_name="data_processing.pathology.refined_table.annotation.generate")
    # disable broadcast join to avoid timeout
    spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "-1")

    # load paths from configs
    bitmask_table_path = const.TABLE_LOCATION(cfg, is_source=True)
    geojson_table_path = const.TABLE_LOCATION(cfg)

    df = spark.read.format("delta").load(bitmask_table_path)

    # explode table by labelsets
    labelsets = get_labelset_keys()
    labelset_column = array([lit(key) for key in labelsets])

    df = df.withColumn("labelset_list", labelset_column)
    # explode labelsets
    df = df.select("slideviewer_path", "slide_id", "sv_project_id", "bmp_record_uuid", "user", "npy_filepath", explode("labelset_list").alias("labelset"))

    # setup variables needed for build geojson UDF
    contour_level = cfg.get_value(path=const.DATA_CFG+'::CONTOUR_LEVEL')
    polygon_tolerance = cfg.get_value(path=const.DATA_CFG+'::POLYGON_TOLERANCE')

    # populate geojson and geojson_record_uuid
    spark.sparkContext.addPyFile("./data_processing/common/EnsureByteContext.py")
    spark.sparkContext.addPyFile("./data_processing/common/utils.py")
    spark.sparkContext.addPyFile("./data_processing/pathology/common/build_geojson.py")
    from build_geojson import build_geojson_from_annotation
    label_config = cfg.get_value(path=const.DATA_CFG+'::LABEL_SETS')

    df = df.withColumn("label_config", lit(str(label_config))) \
            .withColumn("contour_level", lit(contour_level)) \
            .withColumn("polygon_tolerance", lit(polygon_tolerance)) \
            .withColumn("geojson", lit(""))

    print(df.select("label_config").collect())

    df = df.groupby(["bmp_record_uuid", "labelset"]).applyInPandas(build_geojson_from_annotation, schema = df.schema)

    # drop empty geojsons that may have been created
    df = df.filter("geojson != ''")

    # populate uuid
    from utils import generate_uuid_dict
    geojson_record_uuid_udf = udf(generate_uuid_dict, StringType())
    spark.sparkContext.addPyFile("./data_processing/common/EnsureByteContext.py")
    df = df.withColumn("geojson_record_uuid", geojson_record_uuid_udf("geojson", array(lit("SVGEOJSON"), "labelset")))

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

    return exit_code


def create_concat_geojson_table():
    """
    Aggregate geojson features for each labelset, in case there are annotations from multiple users.
    """
    exit_code = 0

    cfg = ConfigSet()
    spark = SparkConfig().spark_session(config_name=const.APP_CFG, app_name="data_processing.pathology.refined_table.annotation.generate")
    spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "false")

    # load paths from config
    geojson_table_path = const.TABLE_LOCATION(cfg, is_source=True)
    concat_geojson_table_path = const.TABLE_LOCATION(cfg)

    concatgeojson_df = spark.read.format("delta").load(geojson_table_path)
    # only use latest annotations for concatenating.
    concatgeojson_df = concatgeojson_df.filter("latest")

    # make geojson string list for slide + labelset
    concatgeojson_df = concatgeojson_df \
        .select("sv_project_id", "slideviewer_path", "slide_id", "labelset", "geojson") \
        .groupby(["sv_project_id", "slideviewer_path", "slide_id", "labelset"]) \
        .agg(collect_list("geojson").alias("geojson_list"))

    # set up udfs
    spark.sparkContext.addPyFile("./data_processing/common/EnsureByteContext.py")
    spark.sparkContext.addPyFile("./data_processing/common/utils.py")
    spark.sparkContext.addPyFile("./data_processing/pathology/common/build_geojson.py")
    from utils import generate_uuid_dict
    from build_geojson import concatenate_regional_geojsons
    concatenate_regional_geojsons_udf = udf(concatenate_regional_geojsons, geojson_struct)
    concat_geojson_record_uuid_udf = udf(generate_uuid_dict, StringType())

    # cache to not have udf called multiple times
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
        windowSpec = Window.partitionBy("slide_id", "labelset").orderBy(desc("date_updated"))
        # Note that last != opposite of first! Have to use desc ordering with first...
        spark.read.format("delta").load(concat_geojson_table_path) \
            .withColumn("date_latest", first("date_updated").over(windowSpec)) \
            .withColumn("latest", col("date_latest")==col("date_updated")) \
            .drop("date_latest") \
            .write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(concat_geojson_table_path)

    logger.info("Finished building Concatenation table.")

    return exit_code


if __name__ == "__main__":
    cli()
