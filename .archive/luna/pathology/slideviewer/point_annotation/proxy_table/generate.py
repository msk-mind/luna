import os
import shutil
import logging

import click
from pyspark.sql.functions import lit, udf, explode, array, to_json
from pyspark.sql.types import (
    ArrayType,
    StringType,
    IntegerType,
    MapType,
    StructType,
    StructField,
)

from luna.common.CodeTimer import CodeTimer
from luna.common.config import ConfigSet
from luna.common.custom_logger import init_logger
from luna.common.sparksession import SparkConfig
from luna.common.utils import get_absolute_path
from luna.pathology.common.slideviewer_client import fetch_slide_ids
import luna.common.constants as const

os.environ["OPENBLAS_NUM_THREADS"] = "1"


def download_point_annotation(slideviewer_url, slideviewer_path, project_id, user):
    """Downloads point-click nuclear annotations using slideviewer API

    Args:
        slideviewer_url (string): slideviewer base url e.g. https://slideviewer-url.com
        slideviewer_path (string): slide path in slideviewer
        project_id (string): slideviewer project id
        user (string): username used to create the expert annotation

    Returns:
        json: point-click nuclear annotations
    """
    from slideviewer_client import download_sv_point_annotation

    print(f" >>>>>>> Processing [{slideviewer_path}] <<<<<<<<")

    url = (
        slideviewer_url
        + "/slides/"
        + str(user)
        + "@mskcc.org/projects;"
        + str(project_id)
        + ";"
        + slideviewer_path
        + "/getSVGLabels/nucleus"
    )
    print(url)

    return download_sv_point_annotation(url)


@click.command()
@click.option(
    "-d",
    "--data_config_file",
    default=None,
    type=click.Path(exists=True),
    help="path to yaml file containing data input and output parameters. "
    "See data_config.yaml.template",
)
@click.option(
    "-a",
    "--app_config_file",
    default="config.yaml",
    type=click.Path(exists=True),
    help="path to yaml file containing application runtime parameters. "
    "See config.yaml.template",
)
def cli(data_config_file, app_config_file):
    """This module generates a parquet table of point-click nuclear annotation jsons.

    The configuration files are copied to your project/configs/table_name folder
    to persist the metadata used to generate the proxy table.

    INPUT PARAMETERS

    app_config_file - path to yaml file containing application runtime parameters. See config.yaml.template

    data_config_file - path to yaml file containing data input and output parameters. See data_config.yaml.template

    - ROOT_PATH: path to output data

    - DATA_TYPE: data type used in table name e.g. POINT_RAW_JSON

    - PROJECT: your project name. used in table path

    - DATASET_NAME: optional, dataset name to version your table

    - PROJECT_ID: Slideviewer project id

    - USERS: list of users that provide expert annotations for this project

    - SLIDEVIEWER_CSV_FILE: an optional path to a SlideViewer csv file to use that lists the names of the whole slide images
    and for which the regional annotation proxy table generator should download point annotations.
    If this field is left blank, then the regional annotation proxy table generator will download this file from SlideViewer.

    TABLE SCHEMA

    - slideviewer_path: path to original slide image in slideviewer platform

    - slide_id: id for the slide. synonymous with image_id

    - sv_project_id: same as the PROJECT_ID from data_config_file, refers to the SlideViewer project number.

    - sv_json: json annotation file downloaded from slideviewer.

    - user: username of the annotator for a given annotation

    - sv_json_record_uuid: hash of raw json annotation file from slideviewer, format: SVPTJSON-{json_hash}
    """
    logger = init_logger()

    with CodeTimer(logger, "generate POINT_RAW_JSON table"):
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

        create_proxy_table()


def create_proxy_table():
    """Create a proxy table of point annotation json files downloaded from the SlideViewer API

    Each row of the table is a point annotation json created by a user for a slide.

    Returns:
        None
    """

    cfg = ConfigSet()
    logger = logging.getLogger(__name__)

    spark = SparkConfig().spark_session(
        config_name=const.APP_CFG,
        app_name="luna.pathology.point_annotation.proxy_table.generate",
    )

    # load paths from configs
    point_table_path = const.TABLE_LOCATION(cfg)

    PROJECT_ID = cfg.get_value(path=const.DATA_CFG + "::PROJECT_ID")
    SLIDEVIEWER_URL = cfg.get_value(path=const.DATA_CFG + "::SLIDEVIEWER_URL")

    # Get slide list to use
    # Download CSV file in the project configs dir
    slides = fetch_slide_ids(
        SLIDEVIEWER_URL,
        PROJECT_ID,
        const.CONFIG_LOCATION(cfg),
        cfg.get_value(path=const.DATA_CFG + "::SLIDEVIEWER_CSV_FILE"),
    )
    logger.info(slides)

    schema = StructType(
        [
            StructField("slideviewer_path", StringType()),
            StructField("slide_id", StringType()),
            StructField("sv_project_id", IntegerType()),
        ]
    )
    df = spark.createDataFrame(slides, schema)
    # populate columns
    df = df.withColumn(
        "users",
        array([lit(user) for user in cfg.get_value(const.DATA_CFG + "::USERS")]),
    )
    df = df.select(
        "slideviewer_path", "slide_id", "sv_project_id", explode("users").alias("user")
    )

    # download slide point annotation jsons
    # example point json:
    # [{"project_id":"8","image_id":"123.svs","label_type":"nucleus","x":"1440","y":"747","class":"0","classname":"Tissue 1"},{"project_id":"8","image_id":"123.svs","label_type":"nucleus","x":"1424","y":"774","class":"3","classname":"Tissue 4"}]
    point_json_struct = ArrayType(MapType(StringType(), StringType()))
    spark.sparkContext.addPyFile(
        get_absolute_path(__file__, "../../../common/slideviewer_client.py")
    )
    download_point_annotation_udf = udf(download_point_annotation, point_json_struct)

    df = df.withColumn(
        "sv_json",
        download_point_annotation_udf(
            lit(SLIDEVIEWER_URL), "slideviewer_path", "sv_project_id", "user"
        ),
    ).cache()
    # drop empty jsons that may have been created
    df = df.dropna(subset=["sv_json"])

    # populate "date_added", "date_updated","latest", "sv_json_record_uuid"
    spark.sparkContext.addPyFile(
        get_absolute_path(__file__, "../../../common/utils.py")
    )
    from luna.common.utils import generate_uuid_dict

    sv_json_record_uuid_udf = udf(generate_uuid_dict, StringType())

    df = df.withColumn(
        "sv_json_record_uuid",
        sv_json_record_uuid_udf(to_json("sv_json"), array(lit("SVPTJSON"))),
    )

    df.show(10, False)
    df.write.format("parquet").mode("overwrite").save(point_table_path)


if __name__ == "__main__":
    cli()
