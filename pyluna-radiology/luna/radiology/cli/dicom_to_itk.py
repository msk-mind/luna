# General imports
import os, logging
import click

from luna.common.custom_logger   import init_logger

init_logger()
logger = logging.getLogger('dicom_to_itk')

from luna.common.utils import cli_runner

_params_ = [('input_data', str), ('output_dir', str), ('itk_image_type', str), ('itk_c_type', str)]

@click.command()
@click.option('-i', '--input_data', required=False,
              help='path to input data')
@click.option('-o', '--output_dir', required=False,
              help='path to output directory to save results')
@click.option('-it', '--itk_image_type', required=False,
              help="repository name to pull model and weight from, e.g. msk-mind/luna-ml")
@click.option('-ct', '--itk_c_type', required=False,
              help="torch hub transform name")   
@click.option('-m', '--method_param_path', required=False,
              help='json file with method parameters for tile generation and filtering')
def cli(**cli_kwargs):
    """
    Run with explicit arguments:

    \b
        infer_tiles
            -i 1412934/data/TileImages
            -o 1412934/data/TilePredictions
            -rn msk-mind/luna-ml:main 
            -tn tissue_tile_net_transform 
            -mn tissue_tile_net_model_5_class
            -wt main:tissue_net_2021-01-19_21.05.24-e17.pth

    Run with implicit arguments:

    \b
        infer_tiles -m 1412934/data/TilePredictions/metadata.json
    
    Run with mixed arguments (CLI args override yaml/json arguments):

    \b
        infer_tiles --input_data 1412934/data/TileImages -m 1412934/data/TilePredictions/metadata.json
    """
    cli_runner(cli_kwargs, _params_, dicom_to_itk)

import itk
def dicom_to_itk(input_data, output_dir, itk_image_type, itk_c_type):
        """
        Generate an ITK compatible image from a dicom series

        :param dicom_path: filepath to folder of dicom images
        :param output_dir: destination directory
        :param params {
            file_ext str: file extention for scan generation
        }

        :return: property dict, None if function fails
        """
        os.makedirs(output_dir, exist_ok=True)

        PixelType = itk.ctype(itk_c_type)
        ImageType = itk.Image[PixelType, 3]

        namesGenerator = itk.GDCMSeriesFileNames.New()
        namesGenerator.SetUseSeriesDetails(True)
        namesGenerator.AddSeriesRestriction("0008|0021")
        namesGenerator.SetGlobalWarningDisplay(False)
        namesGenerator.SetDirectory(input_data)

        seriesUIDs = namesGenerator.GetSeriesUIDs()
        num_dicoms = len(seriesUIDs)

        if num_dicoms < 1:
            logger.warning('No DICOMs in: ' + input_data)
            return None

        logger.info('The directory {} contains {} DICOM Series'.format(input_data, str(num_dicoms)))

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

            outFileName = os.path.join(output_dir, uid + '_volumetric_image.' + itk_image_type)
            writer.SetFileName(outFileName)
            writer.UseCompressionOn()
            writer.SetInput(reader.GetOutput())
            logger.info('Writing: ' + outFileName)
            writer.Update()

        # Prepare metadata and commit
        properties = {
            'output_file+name' : outFileName,
            'num_slices' : n_slices,
        }

        return properties


if __name__ == "__main__":
    cli()
