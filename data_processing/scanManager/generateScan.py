'''
Created: January 2021
@author: aukermaa@mskcc.org

Given a scan (container) ID
1. resolve the path to the dicom folder
2. generate a volumentric image using ITK
3. store results on HDFS and add metadata to the graph

'''

# General imports
import os, json, sys
import click
from checksumdir import dirhash

# From common
from data_processing.common.Neo4jConnection import Neo4jConnection
from data_processing.common.custom_logger   import init_logger
from data_processing.common.Node       import Node

# Specialized library to generate volumentric images
import itk

logger = init_logger("generateScan.log")

def get_container_data(container_id):
    conn = Neo4jConnection(uri=os.environ["GRAPH_URI"], user="neo4j", pwd="password")

    input_nodes = conn.query(f""" MATCH (object:scan)-[:HAS_DATA]-(data:dicom) WHERE id(object)={container_id} RETURN data""")
    
    if not input_nodes or len (input_nodes)==0:
        return [] 
    else:
        return input_nodes[0].data()['data']

def get_method_data(method_id):
    with open(f'{method_id}.json') as json_file:
        method_config = json.load(json_file)['params']
    return method_config

def add_container_data(container_id, n_meta):
    conn = Neo4jConnection(uri=os.environ["GRAPH_URI"], user="neo4j", pwd="password")

    res = conn.query(f""" 
        MATCH (sc:scan) WHERE id(sc)={container_id}
        MERGE (da:{n_meta.get_create_str()})
        MERGE (sc)-[:HAS_DATA]->(da)"""
    )

@click.command()
@click.option('-c', '--cohort_id',    required=True)
@click.option('-s', '--container_id', required=True)
@click.option('-m', '--method_id',    required=True)
def cli(cohort_id, container_id, method_id):
    logger.info("Invocation: " + str(sys.argv))

    conn = Neo4jConnection(uri=os.environ["GRAPH_URI"], user="neo4j", pwd="password")

    properties = {}
    properties['Namespace'] = cohort_id
    properties['MethodID']  = method_id

    input_data  = get_container_data(container_id)
    method_data = get_method_data(method_id) 

    file_ext     = method_data['file_ext']

    logger.info (input_data)
    logger.info (method_data)

    input_dir, filename  = os.path.split(input_data['path'])
    input_dir = input_dir[input_dir.index("/"):]

    output_dir = os.path.join(os.environ['MIND_GPFS_DIR'], "data/COHORTS", cohort_id, "scans", input_data['SeriesInstanceUID'])
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    PixelType = itk.ctype('signed short')
    Dimension = 3

    ImageType = itk.Image[PixelType, Dimension]

    namesGenerator = itk.GDCMSeriesFileNames.New()
    namesGenerator.SetUseSeriesDetails(True)
    namesGenerator.AddSeriesRestriction("0008|0021")
    namesGenerator.SetGlobalWarningDisplay(False)
    namesGenerator.SetDirectory(input_dir)

    seriesUIDs = namesGenerator.GetSeriesUIDs()
    num_dicoms = len(seriesUIDs)

    if num_dicoms < 1:
        logger.warning('No DICOMs in: ' + input_dir)
        exit(1)

    logger.info('The directory {} contains {} DICOM Series: '.format(input_dir, str(num_dicoms)))

    n_slices = 0

    for uid in seriesUIDs:
        logger.info('Reading: ' + uid)
        fileNames = namesGenerator.GetFileNames(uid)
        if len(fileNames) < 1: continue

        n_slices = len(fileNames)

        reader = itk.ImageSeriesReader[ImageType].New()
        dicomIO = itk.GDCMImageIO.New()
        reader.SetImageIO(dicomIO)
        reader.SetFileNames(fileNames)
        reader.ForceOrthogonalDirectionOff()

        writer = itk.ImageFileWriter[ImageType].New()

        outFileName = os.path.join(output_dir, uid + '.' + file_ext)
        writer.SetFileName(outFileName)
        writer.UseCompressionOn()
        writer.SetInput(reader.GetOutput())
        logger.info('Writing: ' + outFileName)
        writer.Update()

    record_name = file_ext + "-" + dirhash(input_dir, "sha256")
    properties['path'] = outFileName
    properties['zdim'] = n_slices

    n_meta = Node(file_ext, record_name, properties=properties)

    add_container_data(container_id, n_meta)

    return 0 


if __name__ == "__main__":
    cli()
