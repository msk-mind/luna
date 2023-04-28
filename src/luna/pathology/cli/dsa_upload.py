import json
import logging

import fire
import girder_client
import requests
from fsspec import open

from luna.common.custom_logger import init_logger
from luna.common.utils import get_config, save_metadata, timed
from luna.pathology.dsa.dsa_api_handler import (
    get_item_uuid,
    push_annotation_to_dsa_image,
    system_check,
)

init_logger()
logger = logging.getLogger("dsa_upload")


@timed
@save_metadata
def cli(
    dsa_endpoint_url: str = "???",
    annotation_file_url: str = "???",
    collection_name: str = "???",
    image_filename: str = "???",
    username: str = "???",
    password: str = "???",
    storage_options: dict = {},
    local_config: str = "",
):
    """Upload annotation to DSA

    Upload json annotation file as a new annotation to the image in the DSA collection.

    Args:
        dsa_endpoint_url (string): DSA endpoint URL
        annotation_file_url (string): URL to a DSA annotation json file
        image_filename (string): name of the image file in DSA e.g. 123.svs
        collection_name (string): name of the collection in DSA
        uri (string): DSA API endpoint e.g. http://localhost:8080/api/v1

    Returns:
        dict: metadata
    """
    config = get_config(vars())
    dsa_uuid = upload_annotation_to_dsa(
        config["dsa_endpoint_url"],
        config["annotation_file_url"],
        config["image_filename"],
        config["username"],
        config["password"],
        config["storage_options"],
    )

    return {"item_uuid": dsa_uuid}


def upload_annotation_to_dsa(
    dsa_endpoint_url: str,
    annotation_file_url: str,
    collection_name: str,
    image_filename: str,
    username: str,
    password: str,
    storage_options: dict = {},
):
    """Upload annotation to DSA

    Upload json annotation file as a new annotation to the image in the DSA collection.

    Args:
        annotation_file_url (string): URL to a DSA annotation json file
        image_filename (string): name of the image file in DSA e.g. 123.svs
        collection_name (string): name of the collection in DSA
        uri (string): DSA API endpoint e.g. http://localhost:8080/api/v1

    Returns:
        dict: item_uuid. None if item doesn't exist
    """
    try:
        gc = girder_client.GirderClient(apiUrl=dsa_endpoint_url)
        # girder python client doesn't support turning off ssl verify.
        # can be removed once we replace the self-signed cert
        session = requests.Session()
        session.verify = False
        gc._session = session
        gc.authenticate(username, password)

        # check DSA connection
        system_check(gc)

    except Exception as exc:
        logger.error(exc)
        raise RuntimeError("Error connecting to DSA API")

    with open(annotation_file_url, **storage_options).open() as annotation_json:
        dsa_annotation = json.load(annotation_json)

    dsa_uuid = get_item_uuid(gc, image_filename, collection_name)

    if dsa_uuid:
        dsa_uuid = push_annotation_to_dsa_image(
            dsa_uuid, dsa_annotation, dsa_endpoint_url[:-6], gc
        )

    return dsa_uuid


if __name__ == "__main__":
    fire.Fire(cli)
