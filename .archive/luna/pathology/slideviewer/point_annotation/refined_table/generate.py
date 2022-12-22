import os
import shutil

import click
from pyspark.sql.functions import lit, udf, explode, array, to_json
from pyspark.sql.types import (
    ArrayType,
    StringType,
    MapType,
    IntegerType,
    StructType,
    StructField,
)

from luna.common.CodeTimer import CodeTimer
from luna.common.config import ConfigSet
from luna.common.custom_logger import init_logger
from luna.common.sparksession import SparkConfig
import luna.common.constants as const
from luna.common.utils import get_absolute_path
from luna.pathology.common.utils import get_labelset_keys

os.environ["OPENBLAS_NUM_THREADS"] = "1"

# geojson struct - example
# [{"type": "Feature", "id": "PathAnnotationObject", "geometry": {"type": "Point", "coordinates": [13981, 15274]}, "properties": {"classification": {"name": "Other"}}},
# {"type": "Feature", "id": "PathAnnotationObject", "geometry": {"type": "Point", "coordinates": [14013, 15279]}, "properties": {"classification": {"name": "Other"}}}]
geojson_struct = ArrayType(
    StructType(
        [
            StructField("type", StringType()),
            StructField("id", StringType()),
            StructField(
                "geometry",
                StructType(
                    [
                        StructField("type", StringType()),
                        StructField("coordinates", ArrayType(IntegerType())),
                    ]
                ),
            ),
            StructField(
                "properties", MapType(StringType(), MapType(StringType(), StringType()))
            ),
        ]
    )
)


@click.command()
@click.option(
    "-d",
    "--data_config_file",
    default=None,
    type=click.Path(exists=True),
    help="path to yaml file containing data input and output parameters. "
    "See ./data_config.yaml.template",
)
@click.option(
    "-a",
    "--app_config_file",
    default="config.yaml",
    type=click.Path(exists=True),
    help="path to yaml file containing application runtime parameters. "
    "See ./app_config.yaml.template",
)
def cli(data_config_file, app_config_file):
    """This module generates a table of point-click nuclear annotations in geojson format.

    This module converts the point annotation jsons in the proxy table to geojson format.
    For more details on point annotation json table, please see `point_annotation/proxy_table/generate.py`

    The configuration files are copied to your project/configs/table_name folder
    to persist the metadata used to generate the proxy table.

    INPUT PARAMETERS

    app_config_file - path to yaml file containing application runtime parameters. See config.yaml.template

    data_config_file - path to yaml file containing data input and output parameters. See data_config.yaml.template

    - ROOT_PATH: path to output data

    - SOURCE_DATA_TYPE: data type specified in proxy table generation e.g. POINT_RAW_JSON

    - DATA_TYPE: data type used in table name e.g. POINT_GEOJSON

    - PROJECT: your project name. used in table path

    - DATASET_NAME: optional, dataset name to version your table

    - LABEL_SETS: annotation label sets defined for this project

    - USE_LABELSET: a labelset name to use within the specified LABEL_SETS. By default uses the 'default_labels' labelset

    - USE_ALL_LABELSETS: True to generate geojsons for all of LABEL_SETS. False to generate geojsons for USE_LABELSET

    TABLE SCHEMA

    - sv_project_id: same as the PROJECT_ID from point annotation proxy data_config_file, refers to the SlideViewer project number.

    - slideviewer_path: path to original slide image in slideviewer platform

    - slide_id: id for the slide. synonymous with image_id

    - sv_json_record_uuid: hash of raw json annotation file from slideviewer, format: SVPTJSON-{json_hash}

    - user: username of the annotator for a given annotation

    - labelset: labelset used to generate geojson

    - geojson: geojson point annotation created

    - geojson_record_uuid: hash of geojson annotation file, format: SVGEOJSON-{labelset}-{geojson_hash}
    """
    logger = init_logger()

    with CodeTimer(logger, "generate GEOJSON table"):
        logger.info("data config file: " + data_config_file)
        logger.info("app config file: " + app_config_file)

        # load configs
        cfg = ConfigSet(name=const.DATA_CFG, config_file=data_config_file)
        cfg = ConfigSet(name=const.APP_CFG, config_file=app_config_file)

        # copy app and data configuration to destination config dir
        config_location = const.CONFIG_LOCATION(cfg)
        os.makedirs(config_location, exist_ok=True)

        shutil.copy(app_config_file, os.path.join(config_location, "app_config.yaml"))
        shutil.copy(data_config_file, os.path.join(config_location, "data_config.yaml"))
        logger.info("config files copied to %s", config_location)

        create_refined_table()


def create_refined_table():
    """Convert point annotation jsons into Geojson format, using the provided labelset mapping.

    Note that the Slideviewer JSON includes a 'classname' field, but this is dependent on the REGIONAL annotation labelset
    that is currently open, which may lead to unexpected behavior.
    This is why the labelset mapping from integer to label is handled downstream in our pipeline versus just using the label from SlideViewer.

    Returns:
        None
    """

    cfg = ConfigSet()
    spark = SparkConfig().spark_session(
        config_name=const.APP_CFG,
        app_name="luna.pathology.point_annotation.refined_table.generate",
    )

    # load paths from configs
    point_table_path = const.TABLE_LOCATION(cfg, is_source=True)
    geojson_table_path = const.TABLE_LOCATION(cfg)

    df = spark.read.format("parquet").load(point_table_path)

    labelsets = get_labelset_keys()
    labelset_names = array([lit(key) for key in labelsets])

    # explode labelsets
    df = df.withColumn("labelset_list", labelset_names).select(
        "slideviewer_path",
        "slide_id",
        "sv_project_id",
        "sv_json_record_uuid",
        "user",
        "sv_json",
        explode("labelset_list").alias("labelset"),
    )
    df.show()

    # build geojsons
    label_config = cfg.get_value(path=const.DATA_CFG + "::LABEL_SETS")

    spark.sparkContext.addPyFile(
        get_absolute_path(__file__, "../../../common/build_geojson.py")
    )
    from build_geojson import build_geojson_from_pointclick_json

    build_geojson_from_pointclick_json_udf = udf(
        build_geojson_from_pointclick_json, geojson_struct
    )
    df = df.withColumn(
        "geojson",
        build_geojson_from_pointclick_json_udf(
            lit(str(label_config)), "labelset", "sv_json"
        ),
    ).cache()

    # populate "date_added", "date_updated","latest", "sv_json_record_uuid"
    spark.sparkContext.addPyFile(
        get_absolute_path(__file__, "../../../common/utils.py")
    )
    from luna.common.utils import generate_uuid_dict

    geojson_record_uuid_udf = udf(generate_uuid_dict, StringType())

    df = df.withColumn(
        "geojson_record_uuid",
        geojson_record_uuid_udf(
            to_json("geojson"), array(lit("SVPTGEOJSON"), "labelset")
        ),
    )

    df.write.format("parquet").mode("overwrite").save(geojson_table_path)


if __name__ == "__main__":
    cli()
