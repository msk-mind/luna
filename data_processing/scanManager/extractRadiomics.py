import os
import itk
import click
from checksumdir import dirhash

from data_processing.common.Neo4jConnection import Neo4jConnection
from data_processing.common.GraphEnum import Node
from data_processing.common.custom_logger import init_logger

logger = init_logger("generateScan.log")

@click.command()
@click.option('-c', '--cohort_id',    required=True)
@click.option('-s', '--container_id', required=True)
@click.option('-m', '--method_id',    required=True)
def cli(cohort_id, container_id, method_id):

    scan_node = conn.query(f"""
        MATCH (object:scan)-[:HAS_DATA]-(image:mhd)
        MATCH (object:scan)-[:HAS_DATA]-(label:mha)
        WHERE id(object)={id}
        RETURN object.cohort, object.AccessionNumber, image.path, label.path"""
    )

    if not scan_node:
        return make_response("Scan is not ready for radiomics (missing annotation?)", 500)

    scan_node = scan_node[0].data()

    JOB_CONFIG = METHODS[method_id]

    config      = JOB_CONFIG['config']
    streams_dir = JOB_CONFIG['streams_dir']
    dataset_dir = JOB_CONFIG['dataset_dir']

    extractor = featureextractor.RadiomicsFeatureExtractor(**config['params'])

    try:
        result = extractor.execute(scan_node["image.path"].split(':')[-1], scan_node["label.path"].split(':')[-1])
    except Exception as e:
        return make_response(str(e), 200)

    sers = pd.Series(result)
    sers["AccessionNumber"] = scan_node["object.AccessionNumber"]
    sers["config"]          = config
    sers.to_frame().transpose().to_csv(os.path.join(streams_dir, id+".csv"))

    with lock:
        if not method_id in STREAMS.keys(): STREAMS[method_id] = START_STREAM(streams_dir, dataset_dir)
        METHODS[method_id]['streamer'] = str(STREAMS[method_id])

    return make_response("Successfully extracted radiomics for case: " + scan_node["object.AccessionNumber"], 200)
