import os
from flask import Flask, request, jsonify, redirect
import girder_client
import logging

from data_processing.common.custom_logger import init_logger

app = Flask(__name__)

logger = init_logger(level=logging.INFO)

DSA_HOSTNAME=os.environ['DSA_HOSTNAME']
DSA_USERNAME=os.environ['DSA_USERNAME'] 
DSA_PASSWORD=os.environ['DSA_PASSWORD']

def dsa_to_geojson(gc, FOLDER_ID, SLIDE_NAME):
    
    item  = next(gc.listItem(folderId=FOLDER_ID, name=SLIDE_NAME))
    annots  = gc.get(f"/annotation?itemId={item['_id']}")
    
    geojson = {
        'type': 'FeatureCollection',
        'features': []
    }
    
    for annot in annots:
        shapes = gc.get(f"/annotation/{annot['_id']}")
        label = shapes['annotation']['name']

        for element in shapes['annotation']['elements']:
            feature = {
                'type': 'Feature',
                'properties': {'label_name' : label},
                'geometry': {
                    'type':'Polygon',
                    'coordinates':[[round(x) for x in p[:2]] for p in element['points']]
                }
            }
            geojson['features'].append(feature)
    return geojson

@app.route('/mind/api/v1/geojson/<folder_id>/<slide_name>', methods=['GET'])
def get_geojson(folder_id, slide_name):
    gc = girder_client.GirderClient(apiUrl=f'http://{DSA_HOSTNAME}/api/v1')
    gc.authenticate(DSA_USERNAME, DSA_PASSWORD)
    return dsa_to_geojson(gc, folder_id, slide_name)

def run():
    app.run(host=os.environ['HOSTNAME'], port=5002, debug=False)


if __name__ == '__main__':
    run()
