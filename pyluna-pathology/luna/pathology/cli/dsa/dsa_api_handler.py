import json, orjson
import requests
import re
import time

from typing import Dict, List, Optional, Tuple
from pathlib import Path


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
        get_collection_id_response = gc.get(
            f"/collection?text={collection_name}"
        )
    except requests.exceptions.HTTPError as err:
        print(
            f"Error in collection id get request: {err.response.status_code}, {err.response.text}"
        )
        return None

    collection_id_dicts = get_collection_id_response

    for collection_id_dict in collection_id_dicts:
        print("collection_id_dict", collection_id_dict)
        if collection_id_dict["name"] == collection_name:
            collection_id = collection_id_dict["_id"]
            print(
                f"Collection {collection_name} found with id: {collection_id}"
            )
            return collection_id

    print(f"Collection {collection_name} not found")
    return None


def get_item_uuid(image_name: str, collection_name: str, gc) -> Optional[str]:
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
        print(
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
    print(f"Image file {image_name} not found")
    return None


def push_annotation_to_dsa_image(
    item_uuid: str, dsa_annotation_json: Dict[str, any], uri: str, gc
):
    """Pushes annotation to DSA, adding given item_uuid (slide-specific id)

    Args:
        item_uuid (str): DSA item uuid to be tied to the annotation
        dsa_annotation_json (Dict[str, any]): annotation JSON in DSA compatable format
        uri (str): DSA host:port e.g. localhost:8080
        gc: girder client

    Returns:
        int: 0 for successful upload, 1 otherwise
    """
    start = time.time()

    # always post a new annotation.
    # updating or deleting an existing annotation for a large annotation document results in timeout.
    try:
        gc.put(
            f"/annotation?itemID={item_uuid}",
            data=orjson.dumps(dsa_annotation_json).decode(),
        )

    except requests.exceptions.HTTPError as err:
        print(
            f"Error in annotation upload: {err.response.status_code}, {err.response.text}"
        )
        return 1

    print("Annotation successfully pushed to DSA.")
    print("Time to push annotation", time.time() - start)
    print(f"http://{uri}/histomics#?image={item_uuid}")
    return 0


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

        print(f"Please check your host or credentials")
        print(err)
        return 1

    print("Successfully connected to DSA")

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
            print(
                f"Error in collection get request: {err.response.status_code}, {err.response.text}"
            )
            return None

        # if response successful, attempt to get stylehseet
        try:
            metadata_stylesheet = collection_response["meta"]["stylesheet"]
        except KeyError:
            print(f"No stylesheet in collection: {collection_uuid}")
            metadata_stylesheet = None
    else:
        print(f"Invalid collection name: {collection_name}")
        return None

    return (collection_uuid, metadata_stylesheet)


def get_slides_from_collection(collection_uuid: str, gc) -> Optional[List[str]]:
    """A helper function that retrieves all slide names from a provided collection via
    accessing DSA resource tree

    Args:
        collection_uuid (str): DSA collection uuid
        gc: girder client
    Returns:
        List[str]: a list of all slide file names belonging to the collection
    """

    # attempt get request for all resources in a collection
    try:
        collection_response = gc.get(
            f"/resource/{collection_uuid}/items?type=collection"
        )
    except requests.exceptions.HTTPError as err:
        print(
            f"Error in collection get request: {err.response.status_code}, {err.response.text}"
        )
        return None

    # getting slide filenames if 'largeImage' key in the collection response

    slide_fnames = [
        resource["name"]
        for resource in collection_response
        if "largeImage" in resource
    ]

    return slide_fnames


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
        slide_id (str): id of WSI on DSA (filename without extension).
        annotation_name (str): name of annotation, or label, created on DSA
        collection_name (str): name of DSA collection the WSI belongs to
        gc: girder client

    Returns:
        Optional[Tuple[str, dict[str, any], dict[str, any]. A tuple consisting of the slide id,
            a json formatted annotation from slideviweer and slide metadata or None if the
            annotation can't be found (ie if image_id, annotation_name or collection_name are
            mis-specified)
    """

    item_uuid = get_item_uuid(slide_id, collection_name, gc)

    # search for annotation

    print("Starting request for annotation")
    try:
        annotation_id_response = gc.get(f"/annotation?itemId={item_uuid}")
        annotation_id = []
        for annot_dict in annotation_id_response:
            try:
                if (
                    annot_dict.get("annotation")
                    and annot_dict.get("annotation").get("name")
                    == annotation_name
                ):
                    annotation_id = annot_dict.get("_id")
                    break
            except AttributeError:
                break

    except requests.exceptions.HTTPError as err:
        print(
            f"Error in annotation get request: {err.response.status_code}, {err.response.text}"
        )
        return None

    if not annotation_id:
        print(
            f"Annotiaton not found for slide: {slide_id} and annotation name: {annotation_name}"
        )
        return None

    try:
        annotation_response = gc.get(f"/annotation/{annotation_id}")
    except requests.exceptions.HTTPError as err:
        print(
            f"Error in annotation get request: {err.response.status_code}, {err.response.text}"
        )
        return None

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
    annotation_name = annotation["name"]    

    try:
        creator_response = gc.get(f"/user/{creator_id}")
    except requests.exceptions.HTTPError as err:
        print(
            f"Error in user get request: {err.response.status_code}, {err.response.text}"
        )
        return None

    creator_login = creator_response["login"]

    try:
        creator_updated_response = gc.get(f"/user/{creator_updated_id}")
    except requests.exceptions.HTTPError as err:
        print(
            f"Error in user get request: {err.response.status_code}, {err.response.text}"
        )
        return None

    creator_login_updated = creator_updated_response["login"]

    slide_metadata = {
        "annotation_name": annotation_name,
        "date": date_created,
        "date_updated": date_updated,
        "user": creator_login,
        "user_updated": creator_login_updated,
    }

    return (slide_id, slide_metadata, json.dumps(annotation))
