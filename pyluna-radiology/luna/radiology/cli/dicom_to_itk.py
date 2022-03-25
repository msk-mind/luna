import os, logging
import click

from luna.common.custom_logger   import init_logger

init_logger()
logger = logging.getLogger('dicom_to_itk')

from luna.common.utils import cli_runner

_params_ = [('input_dicom_folder', str), ('output_dir', str), ('itk_image_type', str), ('itk_c_type', str)]

@click.command()
@click.argument('input_dicom_folder', nargs=1)
@click.option('-o', '--output_dir', required=False,
              help='path to output directory to save results')
@click.option('-it', '--itk_image_type', required=False,
              help="desired ITK image extention")
@click.option('-ct', '--itk_c_type', required=False,
              help="desired C datatype (float, unsigned short)")   
@click.option('-m', '--method_param_path', required=False,
              help='path to a metadata json/yaml file with method parameters to reproduce results')
def cli(**cli_kwargs):
    """Generates a ITK compatible image from a dicom series
    
    \b
    Inputs:
        input_dicom_folder: A folder containing a dicom series
    \b
    Outputs:
        itk_volume
    \b
    Example:
        dicom_to_itk ./scans/10000/2/DICOM/
            --itk_image_type nrrd
            --itk_c_type 'unsigned short'
            -o ./scans/10000/2/NRRD
    """
    cli_runner(cli_kwargs, _params_, dicom_to_itk)

import itk

from pydicom import dcmread
from pathlib import Path

def dicom_to_itk(input_dicom_folder, output_dir, itk_image_type, itk_c_type):
    """Generate an ITK compatible image from a dicom series/folder

    Args:
        input_dicom_folder (str): path to dicom series within a folder
        output_dir (str): output/working directory
        itk_image_type (str): ITK volume image type to output (mhd, nrrd, nii, etc.)
        itk_c_type (str): pixel (C) type for pixels, e.g. float or unsigned short

    Returns:
        dict: metadata about function call
    """
    PixelType = itk.ctype(itk_c_type)
    ImageType = itk.Image[PixelType, 3]

    namesGenerator = itk.GDCMSeriesFileNames.New()
    namesGenerator.SetUseSeriesDetails(True)
    namesGenerator.AddSeriesRestriction("0008|0021")
    namesGenerator.SetGlobalWarningDisplay(False)
    namesGenerator.SetDirectory(input_dicom_folder)

    seriesUIDs = namesGenerator.GetSeriesUIDs()
    num_dicoms = len(seriesUIDs)

    if num_dicoms < 1:
        logger.warning('No DICOMs in: ' + input_dicom_folder)
        return None

    logger.info('The directory {} contains {} DICOM Series'.format(input_dicom_folder, str(num_dicoms)))

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

    path = next(Path(input_dicom_folder).glob("*.dcm"))
    ds = dcmread(path)

    # Prepare metadata and commit
    properties = {
        'itk_volume' : outFileName,
        'num_slices' : n_slices,
        'segment_keys':{
            "radiology_patient_name": str(ds.PatientName),
            "radiology_accession_number": str(ds.AccessionNumber),
            "radiology_series_instance_uuid": str(ds.SeriesInstanceUID),
            "radiology_series_number": str(ds.SeriesNumber),
            "radiology_modality": str(ds.Modality),
        }
    }

    return properties


if __name__ == "__main__":
    cli()

