"""
The steps for processing pathology nuclei point annotations includes:
1) (proxy_table) JSON Anotations are first downloaded from slideviewer
2) (refined_table) JSON annotations are then converted into qupath-compatible geojson files
"""

import os, json
import shutil

import click
from pyspark.sql.window import Window
from pyspark.sql.functions import first, last, col, lit, desc, udf, explode, array, to_json, current_timestamp
from pyspark.sql.types import ArrayType, StringType, IntegerType, MapType, StructType, StructField

from data_processing.common.CodeTimer import CodeTimer
from data_processing.common.config import ConfigSet
from data_processing.common.custom_logger import init_logger
from data_processing.common.sparksession import SparkConfig
from data_processing.pathology.common.slideviewer_client import fetch_slide_ids
import data_processing.common.constants as const

logger = init_logger()

os.environ['OPENBLAS_NUM_THREADS'] = '1'


def download_point_annotation(slideviewer_url, slideviewer_path, project_id, user):
    """
    Return json response from slide viewer call

    :param slideviewer_url: slideviewer base url
    :param slideviewer_path: slide path in slideviewer
    :param project_id: slideviewer project id
    :param user: username
    :return: json response from slideviewer API
    """
    from slideviewer_client import download_sv_point_annotation

    print (f" >>>>>>> Processing [{slideviewer_path}] <<<<<<<<")

    url = slideviewer_url + "/slides/" + str(user) + "@mskcc.org/projects;" + \
          str(project_id) + ';' + slideviewer_path + "/getSVGLabels/nucleus"
    print(url)

    return download_sv_point_annotation(url)


@click.command()
@click.option('-d', '--data_config_file', default=None, type=click.Path(exists=True),
              help="path to yaml file containing data input and output parameters. "
                   "See ./data_config.yaml.template")
@click.option('-a', '--app_config_file', default='config.yaml', type=click.Path(exists=True),
              help="path to yaml file containing application runtime parameters. "
                   "See ./app_config.yaml.template")
def cli(data_config_file, app_config_file):
    """
        This module generates a delta table with point_json_raw pathology data based on the input and output
        parameters specified in the data_config_file.

        Example:
            python3 -m data_processing.pathology.point_annotation.proxy_table.generate \
                     --data_config_file <path to data config file> \
                     --app_config_file <path to app config file>
    """
    with CodeTimer(logger, 'generate POINT_RAW_JSON table'):
        logger.info('data config file: ' + data_config_file)
        logger.info('app config file: ' + app_config_file)

        # load configs
        cfg = ConfigSet(name=const.DATA_CFG, config_file=data_config_file)
        cfg = ConfigSet(name=const.APP_CFG,  config_file=app_config_file)

        # copy app and data configuration to destination config dir
        config_location = const.CONFIG_LOCATION(cfg)
        os.makedirs(config_location, exist_ok=True)

        shutil.copy(app_config_file, os.path.join(config_location, "app_config.yaml"))
        shutil.copy(data_config_file, os.path.join(config_location, "data_config.yaml"))
        logger.info("config files copied to %s", config_location)

        create_proxy_table()


def create_proxy_table():

    cfg = ConfigSet()

    spark = SparkConfig().spark_session(config_name=const.APP_CFG, app_name="data_processing.pathology.point_annotation.proxy_table.generate")

    # load paths from configs
    point_table_path = const.TABLE_LOCATION(cfg)

    PROJECT_ID = cfg.get_value(path=const.DATA_CFG+'::PROJECT_ID')
    SLIDEVIEWER_URL = cfg.get_value(path=const.DATA_CFG+'::SLIDEVIEWER_URL')

    # Get slide list to use
    slides = fetch_slide_ids(SLIDEVIEWER_URL, PROJECT_ID, '.', cfg.get_value(path=const.DATA_CFG+'::SLIDEVIEWER_CSV_FILE'))
    logger.info(slides)

    schema = StructType([StructField("slideviewer_path", StringType()),
                         StructField("slide_id", StringType()),
                         StructField("sv_project_id", IntegerType())
                         ])
    df = spark.createDataFrame(slides, schema)
    # populate columns
    df = df.withColumn("users", array([lit(user) for user in cfg.get_value(const.DATA_CFG+'::USERS')]))
    df = df.select("slideviewer_path", "slide_id", "sv_project_id", explode("users").alias("user"))

    # download slide point annotation jsons
    # example point json:
    # [{"project_id":"8","image_id":"123.svs","label_type":"nucleus","x":"1440","y":"747","class":"0","classname":"Tissue 1"},{"project_id":"8","image_id":"123.svs","label_type":"nucleus","x":"1424","y":"774","class":"3","classname":"Tissue 4"}]
    point_json_struct = ArrayType(
        MapType(StringType(), StringType())
    )
    spark.sparkContext.addPyFile("./data_processing/pathology/common/slideviewer_client.py")
    download_point_annotation_udf = udf(download_point_annotation,  point_json_struct)

    df = df.withColumn("sv_json",
                       download_point_annotation_udf(lit(SLIDEVIEWER_URL), "slideviewer_path", "sv_project_id", "user"))\
        .cache()
    # drop empty jsons that may have been created
    df = df.dropna(subset=["sv_json"])

    # populate "date_added", "date_updated","latest", "sv_json_record_uuid"
    spark.sparkContext.addPyFile("./data_processing/common/EnsureByteContext.py")
    spark.sparkContext.addPyFile("./data_processing/common/utils.py")
    from utils import generate_uuid_dict
    sv_json_record_uuid_udf = udf(generate_uuid_dict, StringType())

    df = df.withColumn("sv_json_record_uuid", sv_json_record_uuid_udf(to_json("sv_json"), array(lit("SVPTJSON")))) \
        .withColumn("latest", lit(True)) \
        .withColumn("date_added", current_timestamp()) \
        .withColumn("date_updated", current_timestamp())

    df.show(10, False)

    # create proxy sv_point json table
    # update main table if exists, otherwise create main table
    if not os.path.exists(point_table_path):
        df.write.format("delta").save(point_table_path)
    else:
        from delta.tables import DeltaTable
        point_table = DeltaTable.forPath(spark, point_table_path)
        point_table.alias("main_point_table") \
            .merge(df.alias("point_annotation_updates"), "main_point_table.sv_json_record_uuid = point_annotation_updates.sv_json_record_uuid") \
            .whenMatchedUpdate(set = { "date_updated" : "point_annotation_updates.date_updated" } ) \
            .whenNotMatchedInsertAll() \
            .execute()

    # add latest flag
    windowSpec = Window.partitionBy("user", "slide_id").orderBy(desc("date_updated"))
    # Note that last != opposite of first! Have to use desc ordering with first...
    spark.read.format("delta").load(point_table_path) \
        .withColumn("date_latest", first("date_updated").over(windowSpec)) \
        .withColumn("latest", col("date_latest")==col("date_updated")) \
        .drop("date_latest") \
        .write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(point_table_path)

if __name__ == "__main__":
    cli()
