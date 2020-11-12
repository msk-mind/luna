#!/gpfs/mskmindhdp_emc/sw/env/bin/python3
from flask import Flask, request
import os

from data_processing.common.CodeTimer import CodeTimer
from data_processing.common.custom_logger import init_logger
from data_processing.common.sparksession import SparkConfig
from data_processing.common.Neo4jConnection import Neo4jConnection

app = Flask(__name__)
logger = init_logger("flask-mind-server.log")

@app.route('/')
def index():
    return "Hello, MSK!"

@app.route('/mind/api/v1/transfer', methods=['POST'])
def transfer():
    config = request.json
    return str(config) 

@app.route('/mind/api/v1/delta', methods=['POST'])
def delta():
    config = request.json
    return str(config) 

@app.route('/mind/api/v1/graph', methods=['POST'])
def graph():
    data = request.json

    spark = SparkConfig().spark_session(os.environ['SPARK_CONFIG'], "data_processing.radiology.proxy_table.generate")

    # Open a connection to the ID graph database
    conn = Neo4jConnection(uri=os.environ["GRAPH_URI"], user="neo4j", pwd="password")

    dicom_header_path = os.path.join(data["TABLE_PATH"], "dicom_header")

    prop_string = ','.join(['''{0}: "{1}"'''.format(prop, data[prop]) for prop in data.keys()])
    conn.query(f'''MERGE (n:dataset{{{prop_string}}})''')

    with CodeTimer(logger, 'setup proxy table'):
        # Reading dicom and opdata
        try:
            df_dcmdata = spark.read.format("delta").load(dicom_header_path)
        except:
            return (f"{dicom_header_path} either does not exist or is not a delta table") 

        tuple_to_add = df_dcmdata.select("PatientName", "SeriesInstanceUID")\
            .groupBy("PatientName", "SeriesInstanceUID")\
            .count()\
            .toPandas()

    with CodeTimer(logger, 'syncronize graph'):

        for index, row in tuple_to_add.iterrows():
            query ='''MATCH (das:dataset {{DATASET_NAME: "{0}"}}) MERGE (px:xnat_patient_id {{value: "{1}"}}) MERGE (sc:scan {{SeriesInstanceUID: "{2}"}}) MERGE (px)-[r1:HAS_SCAN]->(sc) MERGE (das)-[r2:HAS_PX]-(px)'''.format(data['DATASET_NAME'], row['PatientName'], row['SeriesInstanceUID'])
            logger.info (query)
            conn.query(query)
    return (f"Dataset {data['DATASET_NAME']} added successfully!")
if __name__ == '__main__':
    app.run(host = os.environ['HOSTNAME'], debug=True)
