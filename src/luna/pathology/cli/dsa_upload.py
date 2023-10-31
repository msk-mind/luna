import json
import re
from pathlib import Path
from typing import List, Union

import fire
import girder_client
import requests
from fsspec import open
from loguru import logger
from pandera.typing import DataFrame

from luna.common.models import SlideSchema
from luna.common.utils import get_config, save_metadata, timed
from luna.pathology.dsa.dsa_api_handler import (
    get_item_uuid,
    get_slide_annotation,
    push_annotation_to_dsa_image,
    system_check,
)


@timed
@save_metadata
def cli(
    dsa_endpoint_url: str = "???",
    annotation_file_urlpath: str = "",
    annotation_file_list_urlpath: str = "",
    collection_name: str = "???",
    image_filename: str = "",
    username: str = "${oc.env:DSA_USERNAME}",
    password: str = "${oc.env:DSA_PASSWORD}",
    force: bool = False,
    insecure: bool = False,
    storage_options: dict = {},
    local_config: str = "",
):
    """Upload annotation to DSA

    Upload json annotation file as a new annotation to the image in the DSA collection.

    Args:
        dsa_endpoint_url (string): DSA API endpoint e.g. http://localhost:8080/api/v1
        annotation_file_urlpath (string): URL/path to a DSA annotation json file
        annotation_file_list_urlpath (string): URL/path to a DSA annotation json file
        collection_name (string): name of the collection in DSA
        image_filename (string): name of the image file in DSA e.g. 123.svs. If not specified, infer from annotiaton_file_urpath
        username (string): DSA username (defaults to environment variable DSA_USERNAME)
        password (string): DSA password (defaults to environment variable DSA_PASSWORD)
        force (bool): upload even if annotation with same name exists for the slide
        insecure (bool): insecure ssl
        storage_options (dict): options to pass to reading functions
        local_config (string): local config yaml url/path

    Returns:
        dict: metadata
    """
    config = get_config(vars())

    if (
        not config["annotation_file_urlpath"]
        and not config["annotation_file_list_urlpath"]
    ):
        raise fire.core.FireError(
            "Specify either annotation_file_urlpath or annotation_file_list_urlpath"
        )

    annotation_file_urlpaths = []
    if config["annotation_file_urlpath"]:
        annotation_file_urlpaths.append(config["annotation_file_urlpath"])
    if config["annotation_file_list_urlpath"]:
        with open(config["annotation_file_list_urlpath"], "r") as of:
            data = of.read()
            annotation_file_urlpaths += data.split("\n")

    uuids = []
    for idx, annotation_file_urlpath in enumerate(annotation_file_urlpaths):
        logger.info(
            f"Uploading {annotation_file_urlpath}: {idx+1}/{len(annotation_file_urlpaths)}"
        )
        image_filename = config["image_filename"]
        if not image_filename:
            image_filename = Path(annotation_file_urlpath).with_suffix(".svs").name
            image_filename = re.sub(".*_", "", image_filename)
            if not image_filename:
                raise ValueError(
                    f"Unable to infer image_filename from {annotation_file_urlpath}"
                )
            logger.info(f"Image filename inferred as {image_filename}")
        dsa_uuid = _upload_annotation_to_dsa(
            config["dsa_endpoint_url"],
            annotation_file_urlpath,
            config["collection_name"],
            image_filename,
            config["username"],
            config["password"],
            config["force"],
            config["insecure"],
            config["storage_options"],
        )
        logger.info(f"Uploaded item to {dsa_uuid}")
        if dsa_uuid:
            uuids.append(dsa_uuid)

    return {"item_uuids": uuids}


def upload_annotation_to_dsa(
    dsa_endpoint_url: str,
    slide_manifest: DataFrame[SlideSchema],
    annotation_column: str,
    collection_name: str,
    image_filename: str,
    username: str,
    password: str,
    force: bool = False,
    insecure: bool = False,
    storage_options: dict = {},
):
    uuids = []
    for slide in slide_manifest.itertuples(name="Slide"):
        uuids += _upload_annotation_to_dsa(
            dsa_endpoint_url,
            slide[annotation_column],
            collection_name,
            image_filename,
            username,
            password,
            force,
            insecure,
            storage_options,
        )
    return uuids


def _upload_annotation_to_dsa(
    dsa_endpoint_url: str,
    annotation_file_urlpaths: Union[str, List[str]],
    collection_name: str,
    image_filename: str,
    username: str,
    password: str,
    force: bool = False,
    insecure: bool = False,
    storage_options: dict = {},
):
    try:
        gc = girder_client.GirderClient(apiUrl=dsa_endpoint_url)
        # girder python client doesn't support turning off ssl verify.
        # can be removed once we replace the self-signed cert
        session = requests.Session()
        if insecure:
            session.verify = False
        gc._session = session
        gc.authenticate(username, password)

        # check DSA connection
        system_check(gc)

    except Exception as exc:
        logger.error(exc)
        raise RuntimeError("Error connecting to DSA API")

    if type(annotation_file_urlpaths) == str:
        annotation_file_urlpaths = [annotation_file_urlpaths]
    uuids = []
    for annotation_file_urlpath in annotation_file_urlpaths:
        uuids.append(
            __upload_annotation_to_dsa(
                gc,
                dsa_endpoint_url,
                annotation_file_urlpath,
                collection_name,
                image_filename,
                force,
                storage_options,
            )
        )
    return uuids


def __upload_annotation_to_dsa(
    gc: girder_client.GirderClient,
    dsa_endpoint_url: str,
    annotation_file_urlpath: str,
    collection_name: str,
    image_filename: str,
    force: bool = False,
    storage_options: dict = {},
):
    """Upload annotation to DSA

    Upload json annotation file as a new annotation to the image in the DSA collection.

    Args:
        dsa_endpoint_url (string): DSA API endpoint e.g. http://localhost:8080/api/v1
        annotation_file_urlpath (string): URL/path to a DSA annotation json file
        collection_name (string): name of the collection in DSA
        image_filename (string): name of the image file in DSA e.g. 123.svs
        username (string): DSA username
        password (string): DSA password
        storage_options (dict): options to pass to reading functions

    Returns:
        dict: item_uuid. None if item doesn't exist
    """

    with open(annotation_file_urlpath, **storage_options).open() as annotation_json:
        dsa_annotation = json.load(annotation_json)

    if not force:
        slide_annotation = get_slide_annotation(
            image_filename, dsa_annotation["name"], collection_name, gc
        )
        if slide_annotation:
            logger.info(
                f"Found {slide_annotation[1]['annotation_id']}: slide {image_filename} in collection {collection_name} already has an annotation named {dsa_annotation['name']}"
            )
            return slide_annotation[1]["annotation_id"]

    dsa_uuid = get_item_uuid(gc, image_filename, collection_name)

    if dsa_uuid:
        dsa_uuid = push_annotation_to_dsa_image(
            dsa_uuid,
            annotation_file_urlpath,
            dsa_endpoint_url[:-6],
            gc,
            storage_options,
        )

    return dsa_uuid


def fire_cli():
    fire.Fire(cli)


if __name__ == "__main__":
    fire_cli()
