import copy
import json
import logging
import os
import shutil

from datetime import datetime
from typing import Dict
from pathlib import Path

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
        multiple annotations as CONCAT user
    - dsa_json_path: file path to json file downloaded from DSA
    - geojson_path: file path to geojson file converted from DSA json format
    - date_updated: annotation updated time
    - date_created: annotation creation time
    - labelset: name of the provided labelset
    - annotation_name: name of the annotation in DSA
    - annotation_type: annotation type

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
    try:
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
    except KeyError:
        print("Annotation isn't a valid regional annotation")
        return None
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

    output_geojson = []
    try:
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
    except KeyError:
        print("Annotation isn't a valid point annotation")
        return None
    return output_geojson


def generate_geojson(
    dsa_annotation_json: Dict[str, any],
    slide_id: str,
    metadata: Dict[str, any],
    labelset: str,
    slide_store_dir: str,
    data_type: str,
) -> pd.DataFrame:
    """Wrapper function that converts DSA json object to a geojson, saves
    the geojson to the object store then gathers associated metadata for the parquet table

    Args:
        dsa_annotation_json (Dict[str, any]): regional annotation JSON string object from DSA
        slide_id (str): slide id
        metadata (Dict[str, any]): slide metadata
        labelset (str): name of the labelset
        slide_store_dir (str): filepath to slide datastore
        data_type (str): the type of annotation to pull from DSA, cooresponding to either
            regional or point annotations depending on if the DATA_TYPE argument in the input
            yaml config is "REGIONAL_METADATA_RESULTS" or "POINT_GEOJSON" resepectively.
    Returns:
        Dict[str, any]: a dictionary to be saved as in the annotation parquet table
    """
    slide_id = Path(slide_id).stem

    store = DataStore_v2(slide_store_dir) 

    # save dsa annotation json
    dsa_json_path = store.write(
        json.dumps(dsa_annotation_json, indent=4),
        store_id=slide_id,
        namespace_id=metadata["user"],
        data_type=data_type+"_DSA_JSON",
        data_tag=labelset,
    )
   
    # build geojson based on type of annotation (regional or point)
    if data_type == "REGIONAL_METADATA_RESULTS":
        geojson_annotation = regional_json_to_geojson(dsa_annotation_json)
        annotation_type = "RegionalAnnotationJSON"
    else:
        geojson_annotation = point_json_to_geojson(dsa_annotation_json)
        annotation_type = "PointAnnotationJSON"

    # TODO:
    # user field should be derived from metadata, downstream processes
    # requires user field to be CONCAT

    geojson_path = store.write(
        json.dumps(geojson_annotation, indent=4),
        store_id=slide_id,
        namespace_id="CONCAT",
        data_type=data_type,
        data_tag=labelset,
    )

    return {
            "project_name": None,  # gets assigned in outer loop
            "slide_id": slide_id,
            "user": "CONCAT",
            "dsa_json_path": dsa_json_path,
            "geojson_path": geojson_path,
            "date_updated": metadata["date_updated"],
            "date_created": metadata["date"],
            "labelset": labelset,
            "annotation_name": metadata["annotation_name"],
            "annotation_type": annotation_type,
    }


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

    #global params
    #params = cfg.get_config_set("APP_CFG")

    uri = cfg.get_value("DATA_CFG::DSA_URI")
    collection_name = cfg.get_value("DATA_CFG::COLLECTION_NAME")
    annotation_name = cfg.get_value("DATA_CFG::ANNOTATION_NAME")
    landing_path = const.PROJECT_LOCATION(cfg)
    label_set = cfg.get_value("DATA_CFG::LABEL_SETS")
    data_type = cfg.get_value("DATA_CFG::DATA_TYPE")
    dsa_username = cfg.get_value("DATA_CFG::DSA_USERNAME")
    dsa_password = cfg.get_value("DATA_CFG::DSA_PASSWORD")

    dask_threads_per_worker = cfg.get_value("APP_CFG::DASK_THREADS_PER_WORKER")
    dask_n_workers = cfg.get_value("APP_CFG::DASK_N_WORKERS")
    dask_memory_limit = cfg.get_value("APP_CFG::DASK_MEMORY_LIMIT")

    # checking annotation type
    if data_type not in ["REGIONAL_METADATA_RESULTS", "POINT_GEOJSON"]:
        logger.error(
            f"Invalid data type: {data_type}, Expected data types are 'REGIONAL_METADATA_RESULTS' or 'POINT_GEOJSON'"
        )
        quit()

    slide_store_dir = os.path.join(landing_path, "slides")

    table_out_dir = const.TABLE_LOCATION(cfg)
    os.makedirs(table_out_dir, exist_ok=True)
    logger.info(f"Table output directory: {table_out_dir}")

    # instantiate girder client
    gc = girder_client.GirderClient(apiUrl=f"http://{uri}/api/v1")
    gc.authenticate(dsa_username, dsa_password)

    # check DSA connection
    system_check(gc)

    # get collection uuid and stylesheet
    # collection metadata is unused, but could be used to set the labelset
    (collection_uuid, collection_metadata) = get_collection_metadata(
        collection_name, gc
    )
    logger.info("Retrieved collection metadata")

    # get slide names
    slide_fnames = get_slides_from_collection(collection_uuid, gc)

    annotation_data = {
        "project_name": [collection_name] * len(slide_fnames),
        "annotation_name": [annotation_name] * len(slide_fnames),
        "slide_id": [fname for fname in slide_fnames],
    }

    # metadata table
    df = pd.DataFrame.from_dict(annotation_data)

    client = Client(threads_per_worker=dask_threads_per_worker,
			n_workers=dask_n_workers,
			memory_limit=dask_memory_limit)

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
            gc,
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
                data_type,
            )
            geojson_futures.append(geojson_future)

    table_data = []
    for geojson_future in as_completed(geojson_futures):

        try:
            if geojson_future.result() is not None:

                geojson_dict = geojson_future.result()
                geojson_dict["project_name"] = collection_name
                table_data.append(geojson_dict)

                logger.info(
                    f"Annotation for slide {slide_id} generated successfully"
                )

        except:
            logger.warning(
                f"Something wrong with future {geojson_future}, skipping"
            )

    client.shutdown()

    df = pd.DataFrame(table_data)
    df.to_parquet(f"{table_out_dir}/{data_type}.parquet")



if __name__ == "__main__":

    cli()

    pass
