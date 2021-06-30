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
from pyspark.sql.types import ArrayType, StringType, MapType, IntegerType, StructType, StructField

from data_processing.common.CodeTimer import CodeTimer
from data_processing.common.config import ConfigSet
from data_processing.common.custom_logger import init_logger
from data_processing.common.sparksession import SparkConfig
import data_processing.common.constants as const
from data_processing.common.utils import get_absolute_path
from data_processing.pathology.common.utils import get_labelset_keys

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

@click.command()
@click.option('-d', '--data_config_file', default=None, type=click.Path(exists=True),
              help="path to yaml file containing data input and output parameters. "
                   "See ./data_config.yaml.template")
@click.option('-a', '--app_config_file', default='config.yaml', type=click.Path(exists=True),
              help="path to yaml file containing application runtime parameters. "
                   "See ./app_config.yaml.template")
def cli(data_config_file, app_config_file):
    """
        This module generates a table with geojson pathology data based on the input and output parameters
        specified in the data_config_file.

        Example:
            python3 -m data_processing.pathology.point_annotation.refined_table.generate \
                     --data_config_file <path to data config file> \
                     --app_config_file <path to app config file>
    """
    with CodeTimer(logger, 'generate GEOJSON table'):
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

        create_refined_table()


def create_refined_table():

    cfg = ConfigSet()
    spark = SparkConfig().spark_session(config_name=const.APP_CFG, app_name="data_processing.pathology.point_annotation.refined_table.generate")

    # load paths from configs
    point_table_path = const.TABLE_LOCATION(cfg, is_source=True)
    geojson_table_path = const.TABLE_LOCATION(cfg)

    df = spark.read.format("parquet").load(point_table_path)

    labelsets = get_labelset_keys()
    labelset_names = array([lit(key) for key in labelsets])

    # explode labelsets
    df = df.withColumn("labelset_list", labelset_names) \
            .select("slideviewer_path", "slide_id", "sv_project_id", "sv_json_record_uuid", "user", "sv_json", explode("labelset_list").alias("labelset"))
    df.show()

    # build geojsons
    label_config = cfg.get_value(path=const.DATA_CFG+'::LABEL_SETS')

    spark.sparkContext.addPyFile(get_absolute_path(__file__, "../../common/build_geojson.py"))
    from build_geojson import build_geojson_from_pointclick_json
    build_geojson_from_pointclick_json_udf = udf(build_geojson_from_pointclick_json,  geojson_struct)
    df = df.withColumn("geojson", build_geojson_from_pointclick_json_udf(lit(str(label_config)), "labelset", "sv_json")).cache()

    # populate "date_added", "date_updated","latest", "sv_json_record_uuid"
    spark.sparkContext.addPyFile(get_absolute_path(__file__, "../../../common/EnsureByteContext.py"))
    spark.sparkContext.addPyFile(get_absolute_path(__file__, "../../../common/utils.py"))
    from utils import generate_uuid_dict
    geojson_record_uuid_udf = udf(generate_uuid_dict, StringType())

    df = df.withColumn("geojson_record_uuid", geojson_record_uuid_udf(to_json("geojson"), array(lit("SVPTGEOJSON"), "labelset")))

    df.write.format("parquet").mode("overwrite").save(geojson_table_path)

if __name__ == "__main__":
    cli()
