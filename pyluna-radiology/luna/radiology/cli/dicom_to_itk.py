import os, logging
import click

from luna.common.custom_logger   import init_logger

init_logger()
logger = logging.getLogger('dicom_to_itk')

from luna.common.utils import cli_runner

_params_ = [('input_data', str), ('output_dir', str), ('itk_image_type', str), ('itk_c_type', str)]

@click.command()
@click.option('-i', '--input_data', required=False,
              help='path to input data (dicom directory)')
@click.option('-o', '--output_dir', required=False,
              help='path to output directory to save results')
@click.option('-it', '--itk_image_type', required=False,
              help="desired ITK image extention")
@click.option('-ct', '--itk_c_type', required=False,
              help="desired C datatype (float, unsigned short)")   
@click.option('-m', '--method_param_path', required=False,
              help='json file with method parameters for tile generation and filtering')
def cli(**cli_kwargs):
    """
    Generates a ITK compatible image from a dicom series

    \b
        dicom_to_itk
            --input_data ./10000/2/DICOM/
            --itk_image_type nrrd
            --itk_c_type 'unsigned short'
            -o ./10000/2/NRRD
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

