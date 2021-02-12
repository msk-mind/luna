from flask import Flask, request, jsonify, render_template, make_response
from flask_restx import Api, Resource, fields
import subprocess
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds
import pandas as pd
from minio import Minio
import os

from data_processing.api.radiologyPreprocessingLibrary.app.service import dicom_to_binary

# Setup configurations
VERSION      = "branch:"+subprocess.check_output(["git","rev-parse" ,"--abbrev-ref", "HEAD"]).decode('ascii').strip()

# Setup App and Api
app = Flask(__name__)
app.config.from_pyfile('app.cfg', silent=False)

api = Api(app, version=VERSION,
          title='radiologyPreprocessingLibrary',
          description='Preprocessing utility functions for radiology data',
          ordered=True)

object_client = Minio(app.config.get('OBJECT_URI'),
                      access_key=app.config.get('OBJECT_USER'),
                      secret_key=app.config.get('OBJECT_PASSWORD'), secure=False)
# param models
image_params = api.model("DicomToImage",
    {
        "paths": fields.String(description="List of dicom paths", required=True, example=['/path/to/1.dcm','/path/to/2.dcm']),
        "width": fields.String(description="Output image width", required=True, example=512),
        "height": fields.String(description="Output image height", required=True, example=512)
    }
)

# FUTURE: a subsetting function that selects instance # list before this?
@api.route('/radiology/images/<project_id>/<scan_id>',
           methods=['POST'],
           doc={"description": "Dicoms in given scan to Image files."}
)
@api.route('/radiology/images/<project_id>/<scan_id>/<download_path>',
           methods=['GET'],
           doc={"description": "Dicoms in given scan to Image files."}
)
class DicomToImage(Resource):

    @api.doc(
        params={'project_id': 'Project Id',
                'scan_id': 'Scan Identifier',
                'download_path': 'Download Path'},
        responses={200: "Success",
                   400: "Images not found"}
    )
    def get(self, project_id, scan_id, download_path):
        # TODO could be a generic function
        print(project_id)
        print(scan_id)
        print(download_path)

        try:
            object_name = "radiology-images/" + scan_id + ".parquet"
            response = object_client.fget_object(project_id, object_name, download_path)
            print(response.metadata)
            return make_response(f"Downloaded object {project_id}/{object_name} at {download_path}", 200)
        except Exception as ex:
            return make_response(ex, 400)

    @api.doc(
        params={'project_id': 'Project Id',
                'scan_id': 'Scan Identifier'},
        responses={200: "Success",
                   400: "Images already exist"}
    )
    @api.expect(image_params)
    def post(self, project_id, scan_id):

        # get request params
        paths = request.json["paths"]
        width = request.json["width"]
        height = request.json["height"]

        print(paths)
        print(width)
        print(height)
        print(project_id)
        print(scan_id)

        binaries = []
        for path in paths:
            binaries.append(dicom_to_binary(path, int(width), int(height)))

        print(len(binaries))

        # TODO save in parquet
        if not object_client.bucket_exists(project_id):
            object_client.make_bucket(project_id)

        uri = os.path.join(project_id, "radiology-images", scan_id + ".parquet")
        # some minimal parquet schema. save request.json also!
        df = pd.DataFrame({"content": binaries})

        # add post request params
        for key, val in request.json.items():
            df[key] = val

        minio = pa.fs.S3FileSystem(scheme="http",
                                   endpoint_override=app.config.get("OBJECT_URI"),
                                   access_key=app.config.get("OBJECT_USER"),
                                   secret_key=app.config.get("OBJECT_PASSWORD"))

        table = pa.Table.from_pandas(df, preserve_index=False)
        pq.write_table(table, uri, filesystem=minio)

"""
# This should be part of a different microservice.
# TODO service that gets image URI and updates the graph
@api.route('/graphs/<node_type>/<context>/<node_name>',
           methods=['GET', 'DELETE'],
           doc={"description": "Retrieve data from graph."}
)
@api.route('/graphs/<node_type>/<context>/<node_name>/<uri>',
           methods=['POST'],
           doc={"description": "Update graph."}
)
@api.doc(
    params={'node_type': 'node type',
            'context': 'node context',
            'node_name': 'node name or id'},
    responses={200: "Success",
               400: "Nodes already exist"}
)
class UpdateGraph(Resource):

    from data_processing.common.Neo4jConnection import Neo4jConnection

    def __init__(self):
        self.conn = Neo4jConnection(app.config.get('GRAPH_URI'),
                               app.config.get('GRAPH_USER'),
                               app.config.get('GRAPH_PASSWORD'))

    # ex) "scan", "BR_16-512", "scan_id"
    def get(self, node_type, context, node_name):

        try:
            response = self.conn.query(
                f"MATCH (n:{node_type}:globals{{QualifiedPath: '{context}::{node_name}' }}) RETURN n")
            if response:
                return make_response(response, "200")
        except Exception as ex:
            return make_response(ex, 500)

    def post(self, node_type, context, node_name, uri):
        # TODO update graph with parquet metadata and object store URI
        # read parquet metadata and add to node???
        df = dd.read_parquet(uri)
        request_details = df.request.item()

        try:
            self.conn.query(f" 
                MATCH (container:{node_type}) WHERE {node_type}.QualifiedPath={context}::{node_name} 
                MERGE (data:{node_type}_derived_data:globals{{QualifiedPath: '{context}::{Path(uri).stem}', uri: {uri}, {request_details} ) 
                MERGE (container)-[:HAS_DATA]->(data)"
            )
            return make_response("Success", 200)
        except Exception as ex:
            return make_response(ex, 500)
        return
"""
"""
 df = dd.from_pandas(pd.DataFrame(data=d), npartitions=2)
>>> dd.to_parquet(df=df,
...               path='abfs://CONTAINER/FILE.parquet'
...               storage_options={'account_name': 'ACCOUNT_NAME',
...                                'account_key': 'ACCOUNT_KEY'}


to unpack parquet with pyarrow/pandas

# 1. Load downloaded parquet to pandas df
table = pq.read_table('test-download.parquet')
df = table.to_pandas()

# 2. Get the parquet directly from minio, and convert to pandas df
import pyarrow as pa
import pyarrow.dataset as ds

minio = pa.fs.S3FileSystem(scheme="http",
                           endpoint_override=app.config.get("OBJECT_URI"),
                           access_key=app.config.get("OBJECT_USER"),
                           secret_key=app.config.get("OBJECT_PASSWORD"))
                           
dataset = ds.dataset("breast-mri/radiology-images/some-scan-id.parquet", filesystem=minio)
df = dataset.to_table().to_pandas()

... show how we can load binary with Image
# 3. Minio Select API

"""

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5005, threaded=True, debug=False)
