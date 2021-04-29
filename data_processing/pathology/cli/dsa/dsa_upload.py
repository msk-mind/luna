import copy
import click,json

from data_processing.pathology.cli.dsa.dsa_api_handler import get_item_uuid, push_annotation_to_dsa_image, system_check

@click.command()
@click.option('-c', '--config',
              help="json including DSA host, port and token info",
              type=click.Path(exists=True),
              required=True)
@click.option("-d", "--data_config",
              help="path to data config file",
              required=True,
              type=click.Path(exists=True))
def cli(config, data_config):
    """
    DSA Annotation Upload CLI
    """

    with open(config) as config_json:
        dsa_config_dict = json.load(config_json)

    with open(data_config) as data_config_json:
        data_config_dict = json.load(data_config_json)

    # Girder Token can be found in the DSA API Swagger Docs under 'token': (http://{host}:8080/api/v1#!/token/token_currentSession)
    uri = dsa_config_dict["host"] + ":" + dsa_config_dict["port"]
    token = dsa_config_dict["token"]

    # TODO use girder client
    # https://girder.readthedocs.io/en/latest/python-client.html#the-python-client-library

    # check DSA connection
    system_check(uri,token)

    annotation_filepath = data_config_dict['annotation_filepath']
    collection_name = data_config_dict['collection_name']
    image_filename = data_config_dict['image_filename']
    upload_annotation_to_dsa(annotation_filepath, image_filename, collection_name, uri, token)

def upload_annotation_to_dsa(annotation_filepath, image_filename, collection_name, uri, token):
    with open(annotation_filepath) as annotation_json:
        dsa_annotation = json.load(annotation_json)
    
    dsa_uuid = get_item_uuid(image_filename, uri, token, collection_name)
    push_annotation_to_dsa_image(dsa_uuid, dsa_annotation, uri, token)
    
if __name__ == '__main__':
    cli()
