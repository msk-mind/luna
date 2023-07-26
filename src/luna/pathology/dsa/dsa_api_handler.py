import json
import fsspec
from loguru import logger
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import girder_client
import histomicstk
import histomicstk.annotations_and_masks.annotation_and_mask_utils
import orjson
import pandas as pd
import requests


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
        df_collections = pd.DataFrame(gc.listCollection())
        if len(df_collections):
            df_collections = df_collections.set_index("_id")
            df_collections = df_collections.query(f"name=='{collection_name}'")
        logger.debug(f"Found collections {df_collections}")
    except Exception as err:
        logger.error(f"Couldn't retrieve data from DSA: {err}")
        raise RuntimeError("Connection to DSA endpoint failed.")

    # Look for a collection called our collection name
    if len(df_collections) == 0:
        logger.debug(f"No matching collection '{collection_name}'")
        return None

    collection_uuid = df_collections.index.item()

    logger.info(
        f"Found collection id={collection_uuid} for collection={collection_name}"
    )

    return collection_uuid


def create_collection(gc, collection_name: str) -> Optional[str]:
    """
    Creates a dsa collection and returns a collection uuid from the created
    collection on successful creation.

    Args:
        gc: girder client
        collection_name (string): name of the collection

    Returns:
        string: DSA collection uuid. Or an error in the post request.
    """
    try:
        gc.createCollection(collection_name)
        logger.debug(f"Created collection {collection_name}")
        new_collection_id = get_collection_uuid(gc, collection_name)
        logger.debug(f"Collection {collection_name} has id {new_collection_id}")
    except Exception as err:
        logger.error(f"Couldn't create collection {collection_name} : {err}")
        return None

    return new_collection_id


def get_folder_uuid(
    gc, folder_name: str, parent_type: str, parent_id: str
) -> Optional[str]:
    """Returns the DSA folder uuid from the provided `folder_name`

    Args:
        gc: girder client
        folder_name (string): name of the folder in DSA
        parent_type (string): type of the parent container (ie. folder, collection)
        parent_id (string): uuid of the parent container

    Returns:
        string: DSA folder uuid. None if nothing matches the collection name or an
                error in the get request
    """
    try:
        df_folders = pd.DataFrame(gc.listFolder(parent_id, parent_type))
        if len(df_folders):
            df_folders = df_folders.set_index("_id")
            df_folders = df_folders.query(f"name=='{folder_name}'")
        logger.debug(f"Found folders {df_folders}")
    except Exception as err:
        logger.error(f"Couldn't retrieve data from DSA: {err}")
        raise RuntimeError("Connection to DSA endpoint failed.")

    if len(df_folders) == 0:
        logger.debug(f"No matching folders '{folder_name}'")
        return None

    folder_uuid = df_folders.index.item()

    logger.info(f"Found folder id={folder_uuid} for folder={folder_name}")

    return folder_uuid


def create_folder(
    gc, folder_name: str, parent_type: str, parent_id: str
) -> Optional[str]:
    """
    Creates a dsa folder and returns a folder uuid from the created
    folder on successful creation.

    Args:
        gc: girder client
        folder_name (string): name of the folder in DSA
        parent_type (string): type of the parent container (ie. folder, collection)
        parent_id (string): uuid of the parent container

    Returns:
        string: DSA folder uuid. Or an error in the post request.
    """
    try:
        gc.createFolder(parent_id, folder_name, parentType=parent_type)
        logger.debug(f"Created folder {folder_name}")
        new_folder_uuid = get_folder_uuid(gc, folder_name, parent_type, parent_id)
        logger.debug(f"Folder {folder_name} has id {new_folder_uuid}")
    except Exception as err:
        logger.error(f"Couldn't create folder {folder_name} : {err}")
        return None

    return new_folder_uuid


def get_assetstore_uuid(gc, assetstore_name: str) -> Optional[str]:
    """Returns the DSA assetstore uuid from the provided `assetstore_name`

    Args:
        gc: girder client
        assetstore_name (string): name of the assetstore in DSA

    Returns:
        string: DSA assetstore uuid. None if nothing matches the assetstore name or an
                error in the get request
    """
    try:
        df_assetstores = pd.DataFrame(gc.get("assetstore?"))
        if len(df_assetstores):
            df_assetstores = df_assetstores.set_index("_id")
            df_assetstores = df_assetstores.query(f"name=='{assetstore_name}'")
        logger.debug(f"Found assetstores {df_assetstores}")
    except Exception as err:
        logger.error(f"Couldn't retrieve data from DSA: {err}")
        raise RuntimeError("Connection to DSA endpoint failed.")

    if len(df_assetstores) == 0:
        logger.debug(f"No matching assetstore '{assetstore_name}'")
        return None

    assetstore_uuid = df_assetstores.index.item()

    logger.info(
        f"Found assetstore id={assetstore_uuid} for assetstore={assetstore_name}"
    )

    return assetstore_uuid


def create_s3_assetstore(
    gc, name: str, bucket: str, access: str, secret: str, service: str
) -> Optional[str]:
    """
    Creates a s3 assetstore.

    Args:
        gc: girder client
        bucket (string): name of the folder in DSA
        access (string): s3 access ID
        secret (string): s3 password
        service (string) : url of the s3 host

    Returns:
        string: DSA assetstore uuid. Or an error in the post request.
    """
    request_url = (
        f"assetstore?name={name}&type=2&bucket={bucket}&accessKeyId={access}"
        + f"&secret={secret}&service={service}"
    )
    try:
        gc.post(request_url)
        logger.debug(f"Created assetstore {name}")
        new_assetstore_uuid = get_assetstore_uuid(gc, name)
        logger.debug(f"Assetstore {name} has id {new_assetstore_uuid}")
    except Exception as err:
        logger.error(f"Couldn't create assetstore {name}: {err}")
        raise RuntimeError("Unable to create s3 assetstore")

    return new_assetstore_uuid


def import_assetstore_to_folder(
    gc, assetstore_uuid: str, destination_uuid: str
) -> Optional[str]:
    """
    Imports the assetstore to the specified destination folder.

    Args:
        gc: girder client
        assetstore_uuid (string): uuid of the assetstore
        destination_uuid (string): uuid of the destination folder

    Returns:
        None, raises error if post request fails
    """
    request_url = f"assetstore/{assetstore_uuid}/import"
    params = {
        "destinationId": destination_uuid,
        "destinationType": "folder",
        "importPath": "/",
    }
    try:
        gc.post(request_url, parameters=params)
        logger.debug(
            f"Importing from assetstore id {assetstore_uuid}"
            + f"to destination id {destination_uuid}"
        )
    except Exception as err:
        logger.error(f"Couldn't import assetstore id {assetstore_uuid} : {err}")
        raise RuntimeError("Unable to import assetstore to collection")


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
    ].to_list()  # This is the annotation UUID, as a list

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
        uuid_response = gc.get(f'/item?text="{image_id}"')

    except requests.exceptions.HTTPError as err:
        logger.error(
            f"Error in item get request: {err.response.status_code}, {err.response.text}"
        )
        return None

    if uuid_response is not None and len(uuid_response) > 0:
        # multiple entries can come up based on substring matches, return the correct item id by checking name field in dictionary.
        for uuid_response_dict in uuid_response:
            if "name" in uuid_response_dict and "_id" in uuid_response_dict:
                if (
                    uuid_response_dict["name"] == image_name
                    and uuid_response_dict["baseParentId"] == collection_uuid
                ):
                    dsa_uuid = uuid_response_dict["_id"]
                    logger.debug(f"Image file {image_name} found with id: {dsa_uuid}")
                    return dsa_uuid
    logger.warning(f"Image file {image_name} not found")
    return None


def get_item_uuid_by_folder(gc, image_name: str, folder_uuid: str) -> Optional[str]:
    """Returns the DSA item uuid from the provided folder

    Args:
        gc: girder client
        image_name (string): name of the image in DSA e.g. 123.svs
        folder_uuid (string): uuid of parent DSA folder

    Returns:
        string: DSA item uuid. None if nothing matches the folder uuid / image name.
    """
    image_id = Path(image_name).stem
    try:
        uuid_response = gc.get(f'/item?text="{image_id}"')

    except requests.exceptions.HTTPError as err:
        logger.error(
            f"Error in item get request: {err.response.status_code}, {err.response.text}"
        )
        return None

    if uuid_response is not None and len(uuid_response) > 0:
        # multiple entries can come up based on substring matches, return the correct item id by checking name field in dictionary.
        for uuid_response_dict in uuid_response:
            if "name" in uuid_response_dict and "_id" in uuid_response_dict:
                if (
                    uuid_response_dict["name"] == image_name
                    and uuid_response_dict["folderId"] == folder_uuid
                ):
                    dsa_uuid = uuid_response_dict["_id"]
                    logger.debug(f"Image file {image_name} found with id: {dsa_uuid}")
                    return dsa_uuid
    logger.warning(f"Image file {image_name} not found")
    return None


def copy_item(gc, item_id: str, destination_id: str):
    """
    Copies the item to the destination.

    Args:
        gc: girder_client
        item_id (string): uuid of the item to be copied
        destination_id (string): uuid of the destination folder
    """
    request_url = f"item/{item_id}/copy?folderId={destination_id}"
    try:
        gc.post(request_url)
    except Exception as err:
        logger.error(f"Error copying item: {err}")
        raise RuntimeError("Can not copy item")


def push_annotation_to_dsa_image(
    item_uuid: str, annotation_file_urlpath: str, uri: str, gc: girder_client.GirderClient, storage_options: dict = {},
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
        fs, path = fsspec.core.url_to_fs(annotation_file_urlpath, **storage_options)
        size = fs.size(path)
        reference = {
            'identifier': f'{Path(path).stem}-AnnotationFile',
            'itemId': item_uuid
        }
        with fs.open(path) as of:
            gc.uploadFile(
                item_uuid,
                of,
                Path(path).name,
                size,
                reference=orjson.dumps(reference).decode()
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
        logger.debug("retreived collection uuid")

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

    logger.debug("Starting request for annotation")
    try:
        annotation_response = gc.get(f"/annotation?itemId={item_uuid}&name={annotation_name}")

    except Exception as err:
        logger.error(f"Error in annotation get request: {err}")
        return None

    # get annotation json from response
    if annotation_response:
        annotation_response = annotation_response[0]
        annotation = annotation_response['annotation']
    else:
        logger.info(f"No annotation found for slide {slide_id}")
        return None


    # get additional slide-level metadata from response
    date_created = annotation_response["created"]
    date_updated = annotation_response["updated"]

    annotation_id = annotation_response["_id"]
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
        "annotation_id": annotation_id,
        "annotation_name": annotation_name,
        "date": date_created,
        "date_updated": date_updated,
        "user": creator_login,
        "user_updated": creator_login_updated,
    }

    return (slide_id, slide_metadata, json.dumps(annotation))
