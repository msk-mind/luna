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
    properties = {}

    conn = Neo4jConnection(uri=os.environ["GRAPH_URI"], user="neo4j", pwd="password")

    properties = {}
    properties['Namespace'] = cohort_id
    properties['MethodID']  = method_id

    n_method = Node("method",  properties=properties)

    res = conn.query(f"""MATCH (me:{n_method.match()}) RETURN me""")

    if not len(res)==1:
        return "No method namespace found"

    method_config = res[0].data()['me']

    file_ext     = method_config['file_ext']

    input_nodes = conn.query(f""" MATCH (object:scan)-[:HAS_DATA]-(data:metadata) WHERE id(object)={container_id} and data.Type="dcm" RETURN data""")
    
    if not input_nodes: return "Nothing there!"

    input_data = input_nodes[0].data()['data']

    path        = input_data['path']

    input_dir, filename  = os.path.split(path)
    input_dir = input_dir[input_dir.index("/"):]

    output_dir = os.path.join("/gpfs/mskmindhdp_emc/data/COHORTS", cohort_id, "scans", input_data['SeriesInstanceUID'])
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
        return make_response("No DICOMs in: " + input_dir, 500)

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

    properties['RecordID'] = file_ext + "-" + dirhash(input_dir, "sha256")
    properties['Type'] = file_ext
    properties['path'] = outFileName
    properties['zdim'] = n_slices

    n_meta = Node("metadata", properties=properties)

    conn.query(f""" 
        MATCH (sc:scan) WHERE id(sc)={container_id}
        MERGE (da:{n_meta.create()})
        MERGE (sc)-[:HAS_DATA]->(da)"""
    )

    return "Done"


if __name__ == "__main__":
    cli()
