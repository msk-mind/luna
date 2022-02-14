import click
import json
import os
import logging

import girder_client

from luna.pathology.cli.dsa.dsa_api_handler import (
    get_item_uuid,
    push_annotation_to_dsa_image,
    system_check,
)
from luna.common.utils import cli_runner
from luna.common.custom_logger import init_logger

init_logger()
logger = logging.getLogger("dsa_upload")


@click.command()
@click.argument("dsa_endpoint", nargs=1)
@click.option(
    "-c",
    "--collection_name",
    help="name of the collection in DSA",
    required=False,
)
@click.option(
    "-f",
    "--image_filename",
    help="name of the image file in DSA e.g. 123.svs",
    required=False,
)
@click.option(
    "-a",
    "--annotation_filepath",
    help="path to a DSA annotation json file",
    required=False,
)
@click.option(
    "-m",
    "--method_param_path",
    required=False,
    help="path to a metadata json/yaml file with method parameters to "
    "reproduce results",
)
def cli(**cli_kwargs):
    """DSA Annotation Upload CLI

    Example:
        export DSA_USERNAME=username
        export DSA_PASSWORD=password
        dsa_upload http://localhost:8080/dsa/api/v1
            --collection_name tcga-data
            --image_filename 123.svs
            --annotation_name TumorVsOther
    """
    params = [
        ("annotation_filepath", str),
        ("collection_name", str),
        ("image_filename", str),
        ("dsa_endpoint", str),
    ]
    cli_runner(cli_kwargs, params, upload_annotation_to_dsa)


def upload_annotation_to_dsa(
    annotation_filepath, image_filename, collection_name, dsa_endpoint
):
    """Upload annotation to DSA

    Upload json annotation file as a new annotation to the image in the DSA collection.

    Args:
        annotation_filepath (string): path to a DSA annotation json file
        image_filename (string): name of the image file in DSA e.g. 123.svs
        collection_name (string): name of the collection in DSA
        uri (string): DSA API endpoint e.g. http://localhost:8080/api/v1

    Returns:
        dict: item_uuid. None if item doesn't exist
    """
    if os.environ["DSA_USERNAME"] is None or os.environ["DSA_PASSWORD"] is None:
        raise RuntimeError(
            "Please set DSA_USERNAME, DSA_PASSWORD environment " "variable"
        )
    try:
        gc = girder_client.GirderClient(apiUrl=dsa_endpoint)
        gc.authenticate(os.environ["DSA_USERNAME"], os.environ["DSA_PASSWORD"])

        # check DSA connection
        system_check(gc)
    except KeyError as exc:
        logger.error(exc)
        raise KeyError("Set DSA_USERNAME, DSA_PASSWORD environment" " variables")
    except Exception as exc:
        logger.error(exc)
        raise RuntimeError("Error connecting to DSA API")

    with open(annotation_filepath) as annotation_json:
        dsa_annotation = json.load(annotation_json)

    dsa_uuid = get_item_uuid(image_filename, collection_name, gc)

    if dsa_uuid:
        dsa_uuid = push_annotation_to_dsa_image(
            dsa_uuid, dsa_annotation, dsa_endpoint[:-6], gc
        )

    return {"item_uuid": dsa_uuid}


if __name__ == "__main__":
    cli()
