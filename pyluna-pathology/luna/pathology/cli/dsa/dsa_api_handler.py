import json, orjson
import requests
import re
import time

image_id_regex = "(.*).svs"


def get_collection_uuid(uri, token, collection_name):
    """Returns the DSA collection uuid from the provided `collection_name`

    Args:
        uri (string): DSA host:port e.g. localhost:8080
        token (string): DSA token from /token/current HistomicsUI API
        collection_name (string): name of the collection in DSA

    Returns:
        string: DSA collection uuid. None if nothing matches the collection name.
    """
    get_collection_id_url = f"http://{uri}/api/v1/collection?text={collection_name}&limit=5&sort=name&sortdir=1"
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Girder-Token": f"{token}",
    }
    get_collection_id_url_response = requests.get(
        get_collection_id_url, headers=headers
    )
    get_collection_id_response = json.loads(get_collection_id_url_response.text)

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


def get_item_uuid(image_name, uri, token, collection_name):
    """Returns the DSA item uuid from the provided `image_name`

    Args:
        image_name (string): name of the image in DSA e.g. 123.svs
        uri (string): DSA host:port e.g. localhost:8080
        token (string): DSA token from /token/current HistomicsUI API
        collection_name (string): name of the collection in DSA

    Returns:
        string: DSA item uuid. None if nothing matches the collection/image name.
    """
    collection_uuid = get_collection_uuid(uri, token, collection_name)
    if not collection_uuid:
        return None

    image_id = re.search(image_id_regex, image_name).group(1)

    get_dsa_uuid_url = f"http://{uri}/api/v1/item?text={image_id}&limit=50&sort=lowerName&sortdir=1"
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Girder-Token": f"{token}",
    }
    get_dsa_uuid_response = requests.get(get_dsa_uuid_url, headers=headers)
    uuid_response = json.loads(get_dsa_uuid_response.text)

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


def push_annotation_to_dsa_image(item_uuid, dsa_annotation_json, uri, token):
    """Pushes annotation to DSA, adding given item_uuid (slide-specific id)

    Args:
        item_uuid (string): DSA item uuid to be tied to the annotation
        dsa_annotation_json (string):
        uri (string): DSA host:port e.g. localhost:8080
        token (string): DSA token from /token/current HistomicsUI API

    Returns:
        int: 0 for successful upload, 1 otherwise
    """
    start = time.time()

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Girder-Token": f"{token}",
    }

    # always post a new annotation.
    # updating or deleting an existing annotation for a large annotation document results in timeout.
    request_url = f"http://{uri}/api/v1/annotation?itemId={item_uuid}"
    response = requests.post(
        request_url,
        data=orjson.dumps(dsa_annotation_json).decode(),
        headers=headers,
    )

    if response.status_code == 200:
        print("Annotation successfully pushed to DSA.")
        print("Time to push annotation", time.time() - start)
        print(f"http://{uri}/histomics#?image={item_uuid}")
        return 0
    else:
        print(
            "Error in annotation upload:", response.status_code, response.text
        )
        return 1


def system_check(uri, token):
    """Check DSA connection with the given host/token

    Args:
        uri (string): DSA host:port e.g. localhost:8080
        token (string): DSA token from /token/current HistomicsUI API

    Returns:
        int: 0 for successful connection, 1 otherwise
    """
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Girder-Token": f"{token}",
    }
    request_url = f"http://{uri}/api/v1/system/check?mode=basic"
    response = requests.get(request_url, headers=headers)

    if response.status_code == 200:
        print("Successfully connected to DSA.")
        return 0
    else:
        print(f"Please check your host/token.")
        return 1


def get_collection_metadata(
    collection_name: str,
    uri: str,
    token: str,
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


def get_slides_from_collection(
    collection_uuid: str, uri: str, token: str
) -> List[str]:
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
        if "largeImage" in resource
    ]

    return slide_fnames


def get_slide_annotation(
    slide_id: str,
    annotation_name: str,
    collection_name: str,
    uri: str,
    token: str,
) -> Optional[Tuple[str, Dict[str, any], Dict[str, any]]]:
    """A helper function that pulls json annotations along with
    metadata for a particular slide from DSA. Used for both point and regional
    annotation types.

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

    annotation_url = (
        f"http://{uri}/api/v1/annotation/{annotation_id}?sort=_id&sordir=1"
    )

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
