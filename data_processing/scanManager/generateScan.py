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
from data_processing.common.Container  import Container
from data_processing.common.utils      import get_method_data

logger = init_logger("generateScan.log")

# Specialized library to generate volumentric images
import itk


@click.command()
@click.option('-c', '--cohort_id',    required=True)
@click.option('-s', '--container_id', required=True)
@click.option('-m', '--method_id',    required=True)
def cli(cohort_id, container_id, method_id):

    # Eventually these will come from a cfg file, or somewhere else
    container_params = {
        'GRAPH_URI':  os.environ['GRAPH_URI'],
        'GRAPH_USER': "neo4j",
        'GRAPH_PASSWORD': "password"
    }

    # Do some setup
    container   = Container( container_params ).setNamespace(cohort_id).lookupAndAttach(container_id)
    method_data = get_method_data(cohort_id, method_id) 
    input_node  = container.get("dicom", method_data['input_name']) # Only get origional dicoms from

    input_dir = str(input_node.path)

    file_ext     = method_data['file_ext']

    output_dir = os.path.join(os.environ['MIND_GPFS_DIR'], "data", container._namespace_id, container._name, method_id)
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    PixelType = itk.ctype('signed short')
    ImageType = itk.Image[PixelType, 3]

    namesGenerator = itk.GDCMSeriesFileNames.New()
    namesGenerator.SetUseSeriesDetails(True)
    namesGenerator.AddSeriesRestriction("0008|0021")
    namesGenerator.SetGlobalWarningDisplay(False)
    namesGenerator.SetDirectory(input_dir)

    seriesUIDs = namesGenerator.GetSeriesUIDs()
    num_dicoms = len(seriesUIDs)

    if num_dicoms < 1:
        container.logger.warning('No DICOMs in: ' + input_dir)
        exit(1)

    container.logger.info('The directory {} contains {} DICOM Series'.format(input_dir, str(num_dicoms)))

    n_slices = 0

    for uid in seriesUIDs:
        container.logger.info('Reading: ' + uid)
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
        container.logger.info('Writing: ' + outFileName)
        writer.Update()

    # Prepare metadata and commit
    record_type = file_ext
    record_name = method_id
    record_properties = {
        'path' : output_dir,
        'zdim' : n_slices,
        'hash':  dirhash(output_dir, "sha256") 
    }

    output_node = Node(record_type, record_name, record_properties)

    container.add(output_node)
    container.saveAll()

if __name__ == "__main__":
    cli()
