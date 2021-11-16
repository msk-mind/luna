import copy
import json
import logging
import os
import requests
import shutil

from datetime import datetime
from typing import Dict, List, Optional, Tuple

import click
import girder_client
import pandas as pd

from dask.distributed import as_completed, Client

import luna.common.constants as const

from luna.common.DataStore import DataStore_v2
from luna.common.CodeTimer import CodeTimer
from luna.common.config import ConfigSet
from luna.common.custom_logger import init_logger
from luna.pathology.cli.dsa.dsa_api_handler import (
    get_collection_uuid,
    get_item_uuid,
    system_check,
    get_collection_metadata,
    get_slides_from_collection,
    get_slide_annotation,
)


# templates for geoJSON format
GEOJSON_BASE = {"type": "FeatureCollection", "features": []}

GEOJSON_POLYGON = {
    "type": "Feature",
    "properties": {},
    "geometry": {"type": "Polygon", "coordinates": []},
}


@click.command()
@click.option(
    "-d",
    "--data_config_file",
    default=None,
    type=click.Path(exists=True),
    help="path to yaml file containing data input and output parameters. "
    "See dask_data_config.yaml.template",
)
@click.option(
    "-a",
    "--app_config_file",
    default="config.yaml",
    type=click.Path(exists=True),
    help="path to yaml file containing application runtime parameters. "
    "See config.yaml.template",
)
def cli(data_config_file: str, app_config_file: str):
    """This module generates parquets with regional annotation pathology data from DSA

    TABLE SCHEMA
    - project_name: name of DSA collection
    - slide_id: slide id. synonymous with image_id
    - user: username of the annotator for a given annotation. For all slides, we combine
        multiple annotations from
    - geojson_path: file path to  geojson file converted from slideviewer json format
    - date: table creation date
    - labelset: name of the provided labelset

    Args:
        app_config_file (str): path to yaml file containing application runtime parameters.
            See config.yaml.template
        data_config_file (str): path to yaml file containing data input and output parameters.
            See dask_data_config.yaml.template

    Returns:
        None
    """
    logger = init_logger()

    # load configs
    cfg = ConfigSet(name="DATA_CFG", config_file=data_config_file)
    cfg = ConfigSet(name="APP_CFG", config_file=app_config_file)

    with CodeTimer(logger, f"generate DSA annotation geojson table"):
        logger.info("data template: " + data_config_file)
        logger.info("config_file: " + app_config_file)

        # copy app and data configuration to destination config dir
        config_location = const.CONFIG_LOCATION(cfg)
        os.makedirs(config_location, exist_ok=True)

        shutil.copy(
            app_config_file, os.path.join(config_location, "app_config.yaml")
        )
        shutil.copy(
            data_config_file, os.path.join(config_location, "data_config.yaml")
        )
        logger.info("config files copied to %s", config_location)

        generate_annotation_table()

        return


def regional_json_to_geojson(
    dsa_annotation_json: Dict[str, any]
) -> Dict[str, any]:
    """converts DSA regional annotations (JSON) to geoJSON format

    Args:
        dsa_annotation_json (Dict[str, any]): JSON regional annotation object
            pulled from DSA

    Returns:
        Dict[str, any]: geoJSON formatted annotation object
    """
    # load json annotation
    dsa_annotation = json.loads(dsa_annotation_json)

    annotation_geojson = copy.deepcopy(GEOJSON_BASE)

    for element in dsa_annotation.get("elements"):

        label = element["label"]["value"]
        points = element["points"]

        point_list = []
        for p in points:
            point_list.append(p[:2])

        polygon = copy.deepcopy(GEOJSON_POLYGON)

        # no label number here, just labelname
        polygon["properties"]["label_name"] = label
        polygon["geometry"]["coordinates"].append(point_list)
        annotation_geojson["features"].append(polygon)

    return json.dumps(annotation_geojson)


def point_json_to_geojson(
    dsa_annotation_json: Dict[str, any]
) -> Dict[str, any]:
    """converts DSA point annotations (JSON) to geoJSON format

    Args:
        dsa_annotation_json (Dict[str, any]): JSON point annotation object pulled
            from DSA

    Returns:
        Dict[str, any]: geoJSON formatted point annotation object
    """

    dsa_annotation = json.loads(dsa_annotation_json)

    annotation_geojson = copy.deepcopy(geojson_base)

    output_geojson = []
    for element in dsa_annotation["elements"]:
        point = {}
        x = element["center"][0]
        y = element["center"][1]
        label = element["label"]["value"]

        coordinates = [x, y]

        point["type"] = "Feature"
        point["id"] = "PathAnnotationObject"
        point["geometry"] = {"type": "Point", "coordinates": coordinates}
        point["properties"] = {"classification": {"name": label}}
        output_geojson.append(point)

    return output_geojson


def generate_geojson(
    dsa_annotation_json: Dict[str, any],
    slide_id: str,
    metadata: Dict[str, any],
    labelset: str,
    slide_store_dir: str,
    annotation_type: str,
) -> pd.DataFrame:
    """Wrapper function that converts DSA json object to a geojson, saves
    the geojson to the object store then gathers associated metadata for the parquet table

    Args:
        dsa_annotation_json (Dict[str, any]): regional annotation JSON string object from DSA
        slide_id (str): slide id
        metadata (Dict[str, any]): slide metadata
        labelset (str): name of the labelset
        slide_store_dir (str): filepath to slide datastore
        annotation_type (str): the type of annotation to pull from DSA. Either 'regional'
            or 'point.
    Returns:
        pd.DataFrame: a pandas dataframe to be saved as a slice of a regional annotation parquet
            table
    """

    # build geojsoa
    if annotation_type == "regional":
        geojson_annotation = regional_json_to_geojson(dsa_annotation_json)
        data_type = "RegionalAnnotationJSON"
    else:
        geojson_annotation = point_json_to_geojson(dsa_annotation_json)
        data_type = "PointAnnotationJSON"

    # TODO:
    # user field should be derived from metadata, downstream processes
    # requires user field to be CONCAT
    # datetime field uses table generation time, not annotation time

    store = DataStore_v2(slide_store_dir)

    path = store.write(
        json.dumps(geojson_annotation, indent=4),
        store_id=slide_id,
        namespace_id="CONCAT",
        data_type=data_type,
        data_tag=labelset,
    )

    df = pd.DataFrame(
        {
            "project_name": None,  # gets assigned in outer loop
            "slide_id": slide_id,
            "user": "CONCAT",
            "geojson_path": path,
            "date": datetime.now(),
            "labelset": labelset,
            "annotation_type": data_type,
        },
        index=[0],
    )
    return df


def generate_annotation_table() -> None:
    """CLI driver function. provided a collection name and annotation name, this
    method generates the annotation table by first retriving the slides associated
    with the collection along with the collection stylesheet. Then, the process
    of pulling the JSON-formated regional annotations, converting them to geoJSON,
    writing the result to disk and generating the resultant parquet table is parallelized
    via Dask.

    Returns:
        None
    """
    logger = logging.getLogger(__name__)

    cfg = ConfigSet()

    global params
    params = cfg.get_config_set("APP_CFG")

    uri = cfg.get_value("DATA_CFG::DSA_URI")
    collection_name = cfg.get_value("DATA_CFG::COLLECTION_NAME")
    annotation_name = cfg.get_value("DATA_CFG::ANNOTATION_NAME")
    girder_token = cfg.get_value("DATA_CFG::GIRDER_TOKEN")
    landing_path = cfg.get_value("DATA_CFG::LANDING_PATH")
    label_set = cfg.get_value("DATA_CFG::LABEL_SETS")
    annotation_type = cfg.get_value("DATA_CFG::ANNOTATION_TYPE")

    # checking annotation type
    if annotation_type not in ["regional", "point"]:
        logger.error(
            f"Invalid annotation type: {annotation_type}, must be either 'regional' or 'point' "
        )
        quit()

    slide_store_dir = os.path.join(landing_path, "slides")

    table_out_dir = const.TABLE_LOCATION(cfg)
    os.makedirs(table_out_dir, exist_ok=True)
    logger.info(f"Table output directory: {table_out_dir}")

    # check DSA connection
    system_check(uri, girder_token)

    # instantiate girder client
    #gc = girder_client.GirderClient(apiURL=f"https://{uri}/app/v1")
    #gc.authenticate(DSA_USERNAME, DSA_PASSWORD)

    # get collection uuid and stylesheet
    # collection metadata is unused, but could be used to set the labelset
    (collection_uuid, collection_metadata) = get_collection_metadata(
        collection_name, uri, girder_token
    )
    logger.info("Retrieved collection metadata")

    # get slide names
    slide_fnames = get_slides_from_collection(
        collection_uuid, uri, girder_token
    )

    annotation_data = {
        "project_name": [collection_name] * len(slide_fnames),
        "annotation_name": [annotation_name] * len(slide_fnames),
        "slide_id": [fname.strip(".svs.") for fname in slide_fnames],
    }

    # metadata table
    df = pd.DataFrame.from_dict(annotation_data)

    # TODO: pass dask params via dask data config once spark is fully depreciated
    client = Client(threads_per_worker=1, n_workers=25, memory_limit=0.1)

    client.run(init_logger)
    logger.info(client)

    json_futures = []
    geojson_futures = []

    # generate annotation table

    for _, row in df.iterrows():
        json_future = client.submit(
            get_slide_annotation,
            row["slide_id"],
            row["annotation_name"],
            row["project_name"],
            uri,
            girder_token,
        )
        json_futures.append(json_future)

    for json_future in as_completed(json_futures):
        if json_future.result() is not None:

            slide_id, slide_metadata, annotation_json = json_future.result()

            geojson_future = client.submit(
                generate_geojson,
                annotation_json,
                slide_id,
                slide_metadata,
                label_set,
                slide_store_dir,
            )
            geojson_futures.append(geojson_future)

    for geojson_future in as_completed(geojson_futures):

        try:
            if geojson_future.result() is not None:

                geojson_segment_df = geojson_future.result()

                slide_id = geojson_segment_df["slide_id"].values[0]
                geojson_segment_df["project_name"] = collection_name

                geojson_segment_df.to_parquet(
                    f"{table_out_dir}/regional_annot_slice_slide={slide_id}.parquet"
                )
                logger.info(
                    f"Annotation for slide {slide_id} generated successfully"
                )

        except:
            logger.warning(
                f"Something wrong with future {geojson_future}, skipping"
            )

    client.shutdown()


if __name__ == "__main__":

    cli()

    pass
