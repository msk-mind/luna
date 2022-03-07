import json
import orjson
import requests
import time

from typing import Dict, Optional, Tuple
from pathlib import Path

import logging
import girder_client
import pandas as pd

import histomicstk
import histomicstk.annotations_and_masks.annotation_and_mask_utils

logger = logging.getLogger(__name__)


def get_collection_uuid(gc, collection_name: str) -> Optional[str]:
    """Returns the DSA collection uuid from the provided `collection_name`

    Args:
        gc: girder client
        collection_name (string): name of the collection in DSA

    Returns:
        string: DSA collection uuid. None if nothing matches the collection name or an
            error in the get request
    """
    try:
        df_collections = pd.DataFrame(gc.listCollection()).set_index("_id")
        logger.debug(f"Found collections {df_collections}")
    except Exception as err:
        logger.error(f"Couldn't retrieve data from DSA: {err}")
        raise RuntimeError("Connection to DSA endpoint failed.")

    # Look for a collection callled our collection name
    df_collections = df_collections.query(f"name=='{collection_name}'")

    if len(df_collections) == 0:
        logger.error(f"No matching collection '{collection_name}'")
        raise RuntimeError(f"No matching collection '{collection_name}'")

    collection_uuid = df_collections.index.item()

    logger.info(
        f"Found collection id={collection_uuid} for collection={collection_name}"
    )

    return collection_uuid


def get_annotation_uuid(gc, item_id, annotation_name):
    df_annotation_data = pd.DataFrame(gc.get(f"annotation?itemId={item_id}"))

    if len(df_annotation_data) == 0:
        logger.warning(f"No matching annotation '{annotation_name}'")
        return None

    df_annotation_data = df_annotation_data.join(
        df_annotation_data["annotation"].apply(pd.Series)
    )

    # See how many there are, and look for our annotation_name
    logger.info(
        f"Found {len(df_annotation_data)} total annotations: {set(df_annotation_data['name'])}"
    )

    df_annotation_data = df_annotation_data.query(f"name=='{annotation_name}'")

    if len(df_annotation_data) == 0:
        logger.warning(f"No matching annotation '{annotation_name}'")
        return None

    logger.info(f"Found an annotation called {annotation_name}!!!!")  # Found it!

    annotation_id = df_annotation_data.reset_index()[
        "_id"
    ].item()  # This is the annotation UUID

    return annotation_id


def get_item_uuid(gc, image_name: str, collection_name: str) -> Optional[str]:
    """Returns the DSA item uuid from the provided `image_name`

    Args:
        image_name (string): name of the image in DSA e.g. 123.svs
        collection_name (str): name of DSA collection
        gc: girder client

    Returns:
        string: DSA item uuid. None if nothing matches the collection/image name.
    """

    collection_uuid = get_collection_uuid(gc, collection_name)
    if not collection_uuid:
        return None

    image_id = Path(image_name).stem

    try:
        uuid_response = gc.get(f"/item?text={image_id}")

    except requests.exceptions.HTTPError as err:
        logger.error(
            f"Error in item get request: {err.response.status_code}, {err.response.text}"
        )
        return None

    if len(uuid_response) > 0:
        # multiple entries can come up based on substring matches, return the correct item id by checking name field in dictionary.
        for uuid_response_dict in uuid_response:
            if "name" in uuid_response_dict and "_id" in uuid_response_dict:
                if (
                    uuid_response_dict["name"] == image_name
                    and uuid_response_dict["baseParentId"] == collection_uuid
                ):
                    dsa_uuid = uuid_response_dict["_id"]
                    print(f"Image file {image_name} found with id: {dsa_uuid}")
                    return dsa_uuid
    logger.warning(f"Image file {image_name} not found")
    return None


def push_annotation_to_dsa_image(
    item_uuid: str, dsa_annotation_json: Dict[str, any], uri: str, gc
):
    """Pushes annotation to DSA, adding given item_uuid (slide-specific id)

    Args:
        item_uuid (str): DSA item uuid to be tied to the annotation
        dsa_annotation_json (Dict[str, any]): annotation JSON in DSA compatable format
        uri (str): DSA scheme://host:port e.g. http://localhost:8080
        gc: girder client

    Returns:
        int: 0 for successful upload, 1 otherwise
    """
    start = time.time()

    # always post a new annotation.
    # updating or deleting an existing annotation for a large annotation
    # document results in timeout.
    try:
        gc.post(
            f"/annotation?itemId={item_uuid}",
            data=orjson.dumps(dsa_annotation_json).decode(),
        )

    except requests.exceptions.HTTPError as err:
        raise RuntimeError(
            f"Error in annotation upload: {err.response.status_code}, "
            + err.response.text
        )

    logger.info("Annotation successfully pushed to DSA.")
    logger.info(f"Time to push annotation {time.time() - start}")
    logger.info(f"{uri}/histomics#?image={item_uuid}")
    return item_uuid


def dsa_authenticate(gc, username, password):
    """Authenticate girder client

    Args:
        gc: girder client
        username (str): DSA username
        password (str): DSA password
    """
    # Initial connnection
    try:
        gc.authenticate(username, password)
        logger.info(f"Connected to DSA @ {gc.urlBase}")
    except girder_client.AuthenticationError:
        logger.exception("Couldn't authenticate DSA due to AuthenticationError")
        raise RuntimeError("Connection to DSA endpoint failed.")
    except Exception:
        logger.exception("Couldn't authenticate DSA due to some other exception")
        raise RuntimeError("Connection to DSA endpoint failed.")


def system_check(gc):
    """Check DSA connection with the girder client

    Args:
        gc: girder client
    Returns:
        int: 0 for successful connection, 1 otherwise
    """

    try:
        _ = gc.get("/system/check")

    except requests.exceptions.HTTPError as err:

        logger.error("Please check your host or credentials")
        logger.error(err)
        return 1

    logger.info("Successfully connected to DSA")

    return 0


def get_collection_metadata(
    collection_name: str, gc
) -> Optional[Tuple[str, Dict[str, any]]]:
    """A function used to get the stylehseet associated with a DSA collection. The stylesheet
    can store the labels used in the annotation process

    Args:
        collection_name (str): name of DSA collection used to store the slides
        gc: girder client
    Returns:
        Optional[Tuple[str, Dict[str, any]]]: a tuple consisting of the collection uuid
            and thei stylesheet in JSON format or None if no stylesheet is associated
            with the provided collection
    """

    collection_uuid = get_collection_uuid(gc, collection_name)

    if collection_uuid is not None:
        print("retreived collection uuid")

        # try get request from girder
        try:
            collection_response = gc.get(f"/collection/{collection_uuid}")
        except requests.exceptions.HTTPError as err:
            logger.error(
                f"Error in collection get request: {err.response.status_code}, {err.response.text}"
            )
            return None

        # if response successful, attempt to get stylehseet
        try:
            metadata_stylesheet = collection_response["meta"]["stylesheet"]
        except KeyError:
            logger.error(f"No stylesheet in collection: {collection_uuid}")
            metadata_stylesheet = None
    else:
        logger.warning(f"Invalid collection name: {collection_name}")
        return None

    return (collection_uuid, metadata_stylesheet)


def get_slide_df(gc, collection_uuid: str) -> pd.DataFrame:
    """Return slide metadata (largeImage items) for a given colleciton as a dataframe

    Args:
        gc: girder client
        collection_uuid (str): DSA collection uuid
    Returns:
        pd.DataFrame: slide metadata, with slide_id and slide_item_uuid as additional indicies
    """

    try:
        resource_response = gc.listResource(
            f"resource/{collection_uuid}/items", {"type": "collection"}
        )
    except Exception:
        logger.error(
            f"Couldn't retrieve resource data from DSA for {collection_uuid}, perhaps the collection UUID does not exist?"
        )
        raise RuntimeError("Retriving slide data from DSA failed.")

    df_slide_items = pd.DataFrame(resource_response).dropna(
        subset=["largeImage"]
    )  # Get largeImage types from collection items

    # Fill additional metadata based on convention (slide_id)
    df_slide_items["slide_id"] = df_slide_items["name"].apply(
        lambda x: Path(x).stem
    )  # The stem
    df_slide_items["slide_item_uuid"] = df_slide_items["_id"]

    logger.info(f"Found {len(df_slide_items)} slides!")

    return df_slide_items


def get_annotation_df(gc, annotation_uuid):
    """Return annotation metadata (regions) for a given annotation as a dataframe

    Args:
        gc: girder client
        annotation_uuid (str): DSA annotation uuid
    Returns:
        pd.DataFrame: annotation/region metadata, with slide_item_uuid as additional indicies
    """
    # Here we get all the annotation data as a json document
    annot = gc.get(f"annotation/{annotation_uuid}")
    (
        df_summary,
        df_regions,
    ) = histomicstk.annotations_and_masks.annotation_and_mask_utils.parse_slide_annotations_into_tables(
        [annot]
    )

    # Lets process the coordiates a bit...
    df_regions["x_coords"] = [
        [int(x) for x in coords_x.split(",")] for coords_x in df_regions["coords_x"]
    ]
    df_regions["y_coords"] = [
        [int(x) for x in coords_x.split(",")] for coords_x in df_regions["coords_y"]
    ]
    df_regions = df_regions.drop(columns=["coords_x", "coords_y"])

    # And join the summary data with the regional data
    df_annotations = (
        df_summary.set_index("annotation_girder_id")
        .join(df_regions.set_index("annotation_girder_id"))
        .reset_index()
    )

    df_annotations = df_annotations.rename(columns={"itemId": "slide_item_uuid"})

    return df_annotations


def get_slide_annotation(
    slide_id: str,
    annotation_name: str,
    collection_name: str,
    gc,
) -> Optional[Tuple[str, Dict[str, any], Dict[str, any]]]:
    """A helper function that pulls json annotations along with
    metadata for a particular slide from DSA. Used for both point and regional
    annotation types.

    Args:
        slide_id (str): image name of WSI on DSA.
        annotation_name (str): name of annotation, or label, created on DSA
        collection_name (str): name of DSA collection the WSI belongs to
        gc: girder client

    Returns:
        Optional[Tuple[str, dict[str, any], dict[str, any]. A tuple consisting of the slide id,
            a json formatted annotation from slideviweer and slide metadata or None if the
            annotation can't be found (ie if image_id, annotation_name or collection_name are
            mis-specified)
    """

    item_uuid = get_item_uuid(gc, slide_id, collection_name)

    # search for annotation

    print("Starting request for annotation")
    try:
        annotation_id_response = gc.get(f"/annotation?itemId={item_uuid}")
        annotation_id = None
        for annot_dict in annotation_id_response:
            try:
                if (
                    annot_dict.get("annotation")
                    and annot_dict.get("annotation").get("name") == annotation_name
                ):
                    annotation_id = annot_dict.get("_id")
                    annotation_response = gc.get(f"/annotation/{annotation_id}")

                    break
            except AttributeError:
                break

    except Exception as err:
        logger.error(f"Error in annotation get request: {err}")
        return None

    # get annotation json from response
    if "annotation" in annotation_response:
        annotation = annotation_response["annotation"]
    else:
        logger.error(f"No annotation found for slide {slide_id}")
        return None

    # get additional slide-level metadata from response
    date_created = annotation_response["created"]
    date_updated = annotation_response["updated"]

    creator_id = annotation_response["creatorId"]
    creator_updated_id = annotation_response["updatedId"]
    annotation_name = annotation["name"]

    try:
        creator_response = gc.get(f"/user/{creator_id}")
        creator_updated_response = gc.get(f"/user/{creator_updated_id}")
    except requests.exceptions.HTTPError as err:
        logger.error(
            f"Error in user get request: {err.response.status_code}, {err.response.text}"
        )
        return None

    creator_login = creator_response["login"]
    creator_login_updated = creator_updated_response["login"]

    slide_metadata = {
        "annotation_name": annotation_name,
        "date": date_created,
        "date_updated": date_updated,
        "user": creator_login,
        "user_updated": creator_login_updated,
    }

    return (slide_id, slide_metadata, json.dumps(annotation))
