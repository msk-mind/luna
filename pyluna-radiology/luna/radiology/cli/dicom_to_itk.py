import os, logging
import click

from luna.common.custom_logger   import init_logger

init_logger()
logger = logging.getLogger('dicom_to_itk')

from luna.common.utils import cli_runner

_params_ = [('input_dicom_folder', str), ('output_dir', str), ('itk_image_type', str), ('convert_to_suv', bool), ('itk_c_type', str)]

@click.command()
@click.argument('input_dicom_folder', nargs=1)
@click.option('-o', '--output_dir', required=False,
              help='path to output directory to save results')
@click.option('-it', '--itk_image_type', required=False,
              help="desired ITK image extention")
@click.option('-ct', '--itk_c_type', required=False,
              help="desired C datatype (float, unsigned short)")   
@click.option('-suv', '--convert_to_suv', required=False,
              help="If an applicable PET image, convert to SUVs", is_flag=True)   
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
import medpy.io

def dicom_to_itk(input_dicom_folder, output_dir, itk_image_type, itk_c_type, convert_to_suv):
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

    if convert_to_suv: convert_pet_volume_to_suv(input_dicom_folder, outFileName )

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

def check_pet(dcms):
    """ Ensures dicom directory contains PET images
    
    Args:
        dcms (list[str]): list of dicom paths

    Returns:
        bool: True if all dicoms are PT modality
    """
    for dcm in dcms:
        ds = dcmread(dcm)
        if not ds.Modality=='PT': 
            logger.warning("check_pet - FAILED - Trying to apply PT corrections to non-PT image, gracefully skipping!")
            return False
    
    logger.info("check_pet - PASSED - These are PT images")

    return True

def check_delay_correction(dcms):
    """ Ensures all dicom images were delay corrected to their Aquisition TIme
    
    Args:
        dcms (list[str]): list of dicom paths
    """
    for dcm in dcms:
        ds = dcmread(dcm)
        if not ds.DecayCorrection=='START': 
            logger.error("check_delay_correction - FAILED - Cannot process a PET volume not constructed of 'START' time delay corrected slices!")
            raise RuntimeError("Cannot process a PET volume not constructed of 'START' time delay corrected slices!")

    logger.info("check_delay_correction - PASSED - All slices were decay corrected to their START AquisitionTime")


def calculate_normalization(dcms):
    """ Calculates the SUV normalization (g/BQ)
    
    Args:
        dcms (list[str]): list of dicom paths

    Returns:
        float: Normalization in (g/BQ)
    """
    logger.info("About to convert PET volume to SUV units")

    ds = dcmread ( dcms[0] ) 

    dose   = float( ds.RadiopharmaceuticalInformationSequence[0].RadionuclideTotalDose )
    weight = float (ds.PatientWeight) * 1000

    logger.info(f"Radionuclide dose={dose}, patient weight={weight}")

    norm = weight / dose

    return norm

def convert_pet_volume_to_suv(input_dicom_folder, input_volume):
    """ Renormalizes a PET ITK volume to SUVs, saves a new volume in-place
    
    Args:
        input_dicom_folder (str): path to matching dicom series within a folder
        input_volume (str): path to PT volume
    """
    dcms = list ( Path(input_dicom_folder).glob('*.dcm') )

    if check_pet(dcms):
        check_delay_correction(dcms)

        norm = calculate_normalization(dcms)
        logger.info(f"Calculated SUV normalization factor={norm}")

        image, header = medpy.io.load(input_volume)

        image = image * norm
        logger.info(f"SUM SUV={image.sum()}")

        medpy.io.save(image, input_volume, hdr=header)

        logger.info(f"Saved normalized volume: {input_volume}")


if __name__ == "__main__":
    cli()
