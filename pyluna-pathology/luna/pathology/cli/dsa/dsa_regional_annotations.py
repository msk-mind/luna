import copy
import json
import logging
import os
import requests
import shutil

from datetime import datetime
from typing import Dict, List, Optional, Tuple

import click
import pandas as pd

from dask.distributed import as_completed, Client

import luna.common.constants as const

from luna.common.DataStore import DataStore_v2
from luna.common.CodeTimer import CodeTimer
from luna.common.config import ConfigSet
from luna.common.custom_logger import init_logger
from luna.pathology.cli.dsa.dsa_api_handler import get_collection_uuid, get_item_uuid, system_check


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

        shutil.copy(app_config_file, os.path.join(config_location, "app_config.yaml"))
        shutil.copy(data_config_file, os.path.join(config_location, "data_config.yaml"))
        logger.info("config files copied to %s", config_location)

        generate_annotation_table()

        return


def get_slide_annotation(
    slide_id: str, annotation_name: str, collection_name: str, uri: str, token: str
) -> Optional[Tuple[str, Dict[str, any], Dict[str, any]]]:
    """A helper function that pulls json annotations along with
    metadata for a particular slide from DSA.

    Args:
        slide_id (str): id of WSI on DSA (filename without extension). assumes .svs format
        annotation_name (str): name of annotation, or label, created on DSA
        collection_name (str): name of DSA collection the WSI belongs to
        uri (str): DSA uri
        token (str): girder API token

    Returns:
        Optional[Tuple[str, dict[str, any], dict[str, any]. A tuple consisting of the slide id,
            a json formatted annotation from slideviweer and slide metadata or None if the
            annotation can't be found (ie if image_id, annotation_name or collection_name are
            mis-specified)
    """

    item_uuid = get_item_uuid(slide_id + ".svs", uri, token, collection_name)

    annotation_url = f"http://{uri}/api/v1/annotation?itemId={item_uuid}&name={annotation_name}&limit=50&sort=lowerName&sortdir=1"

    header = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Girder-Token": f"{token}",
    }

    # search for annotation
    print("Starting request for annotation")
    response = requests.get(annotation_url, headers=header)

    annotation_id_response = json.loads(response.text)
    annotation_id = []
    for annot_dict in annotation_id_response:
        try:
            if (
                annot_dict.get("annotation")
                and annot_dict.get("annotation").get("name") == annotation_name
            ):
                annotation_id = annot_dict.get("_id")
                break
        except AttributeError:
            break

    if annotation_id is None:
        print(
            f"Annotiaton not found for slide: {slide_id} and annotation name: {annotation_name}"
        )
        return None

    annotation_url = f"http://{uri}/api/v1/annotation/{annotation_id}?sort=_id&sordir=1"

    response = requests.get(annotation_url, headers=header)

    annotation_response = json.loads(response.text)

    # get annotation json from response
    try:
        annotation = annotation_response["annotation"]
    except KeyError:
        print(f"No annotation found for slide {slide_id}")
        return None

    # get additional slide-level metadata from response
    date_created = annotation_response["created"]
    date_updated = annotation_response["updated"]

    creator_id = annotation_response["creatorId"]
    creator_updated_id = annotation_response["updatedId"]

    # query for creator logins
    creator_response = requests.get(
        f"http://{uri}/api/v1/user/{creator_id}", headers=header
    )
    creator_response = json.loads(creator_response.text)
    creator_login = creator_response["login"]

    creator_updated_response = requests.get(
        f"http://{uri}/api/v1/user/{creator_updated_id}", headers=header
    )
    creator_updated_response = json.loads(creator_updated_response.text)
    creator_login_updated = creator_updated_response["login"]

    slide_metadata = {
        "date": date_created,
        "date_updated": date_updated,
        "user": creator_login,
        "user_updated": creator_login_updated,
    }

    return (slide_id, slide_metadata, json.dumps(annotation))


def get_slides_from_collection(collection_uuid: str, uri: str, token:str) -> List[str]:
    """A helper function that retrieves all slide names from a provided collection via
    accessing DSA resource tree

    Args:
        collection_uuid (str): DSA collection uuid
        uri (str): DSA uri
        token (str): girder client token
    Returns:
        List[str]: a list of all slide file names belonging to the collection
    """
    # get request for all resources (folder, collection, user) that are
    # children of the collection
    request_url = f"http://{uri}/api/v1/resource/{collection_uuid}/items?type=collection&limit=1000&sort=_id&sortdir=1"
    
    header = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Girder-Token": f"{token}",
    }

    response = requests.get(request_url, headers=header)

    collection_response = json.loads(response.text)

    # getting slide filenames if 'largeImage' key in the response
    slide_fnames = [
        resource["name"]
        for resource in collection_response
        if 'largeImage' in resource
    ]

    return slide_fnames


def get_collection_metadata(
    collection_name: str, uri: str, token:str,
) -> Optional[Tuple[str, Dict[str, any]]]:
    """A function used to get the stylehseet associated with a DSA collection. The stylesheet
    can store the labels used in the annotation process

    Args:
        collection_name (str): name of DSA collection used to store the slides
        uri (str): DSA uri
        token (str): girder client token
    Returns:
        Optional[Tuple[str, Dict[str, any]]]: a tuple consisting of the collection uuid
            and thei stylesheet in JSON format or None if no stylesheet is associated
            with the provided collection
    """

    collection_uuid = get_collection_uuid(uri, token, collection_name)
    
    header = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Girder-Token": f"{token}",
    }


    if collection_uuid is not None:
        print("retreived collection uuid")
        request_url = f"http://{uri}/api/v1/collection/{collection_uuid}"
        
        response = requests.get(request_url, headers=header)
        collection_response = json.loads(response.text)
        try:
            metadata_stylesheet = collection_response["meta"]["stylesheet"]
        except KeyError:
            print("No stylesheet in the collection")
            metadata_stylesheet = None
    else:
        print("Invalid collection uuid")
        return None

    return (collection_uuid, metadata_stylesheet)


def regional_json_to_geojson(dsa_annotation_json: Dict[str, any]) -> Dict[str, any]:
    """converts DSA regional annotations (JSON) to geoJSON format

    Args:
        dsa_annotation_json (Dict[str, any]): JSON annotation object pulled from DSA

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


def generate_geojson(
    dsa_annotation_json: Dict[str, any],
    slide_id: str,
    metadata: Dict[str, any],
    labelset: str,
    slide_store_dir: str,
) -> pd.DataFrame:
    """Wrapper function that converts DSA json object to a geojson, saves
    the geojson to the object store then gathers associated metadata for the parquet table

    Args:
        dsa_annotation_json (Dict[str, any]): regional annotation JSON string object from DSA
        slide_id (str): slide id
        metadata (Dict[str, any]): slide metadata
        labelset (str): name of the labelset
        slide_store_dir (str): filepath to slide datastore
    Returns:
        pd.DataFrame: a pandas dataframe to be saved as a slice of a regional annotation parquet
            table
    """

    # build geojson
    geojson_annotation = regional_json_to_geojson(dsa_annotation_json)

    # TODO:
    # user field should be derived from metadata, downstream processes
    # requires user field to be CONCAT
    # datetime field uses table generation time, not annotation time

    store = DataStore_v2(slide_store_dir)

    path = store.write(
        json.dumps(geojson_annotation, indent=4),
        store_id=slide_id,
        namespace_id="CONCAT",
        data_type="RegionalAnnotationJSON",
        data_tag=labelset,
    )

    df = pd.DataFrame(
        {
            "project_name": None, # gets assigned in outer loop
            "slide_id": slide_id,
            "user": "CONCAT",
            "geojson_path": path,
            "date": datetime.now(),
            "labelset": labelset,
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

    slide_store_dir = os.path.join(landing_path, "slides")

    table_out_dir = const.TABLE_LOCATION(cfg)
    os.makedirs(table_out_dir, exist_ok=True)
    logger.info(f"Table output directory: {table_out_dir}")

    # check DSA connection
    system_check(uri, girder_token)

    # get collection uuid and stylesheet
    # collection metadata is unused, but could be used to set the labelset
    (collection_uuid, collection_metadata) = get_collection_metadata(
        collection_name, uri, girder_token
    )
    logger.info("Retrieved collection metadata")

    # get slide names
    slide_fnames = get_slides_from_collection(collection_uuid, uri, girder_token)

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
                slide_store_dir
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
                logger.info(f"Annotation for slide {slide_id} generated successfully")

        except:
            logger.warning(f"Something wrong with future {geojson_future}, skipping")

    client.shutdown()


if __name__ == "__main__":

    cli()

    pass
