import json, orjson
import requests
import re
import time

image_id_regex = "(.*).svs"


def get_item_uuid(image_name, uri, token):
    """
    Returns the DSA item uuid from the provided image_name
    Expects image_name to be the image name with svs extension.
    """
    image_id = re.search(image_id_regex, image_name).group(1)

    get_dsa_uuid_url = f"http://{uri}/api/v1/item?text={image_id}&limit=50&sort=lowerName&sortdir=1"
    headers = {'Content-Type': 'application/json', 'Accept': 'application/json', 'Girder-Token': f'{token}'}
    get_dsa_uuid_response = requests.get(get_dsa_uuid_url, headers=headers)
    uuid_response = json.loads(get_dsa_uuid_response.text) 
    
    if len(uuid_response) > 0 :
        # multiple entries can come up based on substring matches, return the correct item id by checking name field in dictionary.
        for uuid_response_dict in uuid_response:
            if "name" in uuid_response_dict and "_id" in uuid_response_dict:
                if uuid_response_dict["name"] == image_name:
                    dsa_uuid =  uuid_response_dict['_id']
                    return dsa_uuid
    return None


def push_annotation_to_dsa_image(item_uuid, dsa_annotation_json, uri, token):
    """
    Pushes DSA annotation to DSA, adding given item_uuid (slide-specific id)
    """
    start = time.time()

    headers = {'Content-Type': 'application/json', 'Accept': 'application/json', 'Girder-Token': f'{token}'}

    # always post a new annotation.
    # updating or deleting an existing annotation for a large annotation document results in timeout.
    request_url = f"http://{uri}/api/v1/annotation?itemId={item_uuid}"
    response = requests.post(request_url, data=orjson.dumps(dsa_annotation_json).decode(), headers=headers)

    if response.status_code == 200:
        print("Annotation successfully pushed to DSA.")
        print("Time to push annotation", time.time() - start)
        print(f"http://{uri}/histomics#?image={item_uuid}")
        return 0
    else:
        print("Error in annotation upload:", response.status_code, response.reason)
        return 1



def system_check(uri, token):
    """
    Check DSA connection with the given host/token

    :param host: DSA host
    :param token: DSA token
    :return: none
    """
    headers = {'Content-Type': 'application/json', 'Accept': 'application/json', 'Girder-Token': f'{token}'}
    request_url = f"http://{uri}/api/v1/system/check?mode=basic"
    response = requests.get(request_url, headers=headers)

    if response.status_code == 200:
        print("Successfully connected to DSA.")
        return 0
    else:
        print(f"Please check your host/token.")
        return 1

