import os
from pathlib import Path

import fire
import itk
import medpy.io
import numpy as np
from loguru import logger
from pydicom import dcmread


def dicom_to_itk(
    dicom_urlpath, output_urlpath, itk_image_type, itk_c_type, convert_to_suv=False
):
    """Generate an ITK compatible image from a dicom series/folder

    Args:
        dicom_urlpath (str): path to dicom series within a folder
        output_urlpath (str): output/working directory
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
    namesGenerator.SetDirectory(dicom_urlpath)

    seriesUIDs = namesGenerator.GetSeriesUIDs()
    num_dicoms = len(seriesUIDs)

    if num_dicoms < 1:
        logger.warning("No DICOMs in: " + dicom_urlpath)
        return None

    logger.info(
        "The directory {} contains {} DICOM Series".format(
            dicom_urlpath, str(num_dicoms)
        )
    )

    n_slices = 0
    volume = {}
    for uid in seriesUIDs:
        logger.info("Reading: " + uid)
        fileNames = namesGenerator.GetFileNames(uid)
        if len(fileNames) < 1:
            continue

        n_slices = len(fileNames)

        reader = itk.ImageSeriesReader[ImageType].New()
        dicomIO = itk.GDCMImageIO.New()
        reader.SetImageIO(dicomIO)
        reader.SetFileNames(fileNames)
        reader.ForceOrthogonalDirectionOff()

        writer = itk.ImageFileWriter[ImageType].New()

        outFileName = os.path.join(
            output_urlpath, uid + "_volumetric_image." + itk_image_type
        )
        writer.SetFileName(outFileName)
        writer.UseCompressionOn()
        writer.SetInput(reader.GetOutput())
        logger.info("Writing: " + outFileName)
        writer.Update()

        img, _ = medpy.io.load(outFileName)
        volume[outFileName] = np.prod(img.shape)

    if convert_to_suv:
        convert_pet_volume_to_suv(dicom_urlpath, outFileName)

    path = next(Path(dicom_urlpath).glob("*.dcm"))
    ds = dcmread(path)

    # If there are multiple seriesUIDs in a single DICOM dir, return
    # the largest one by volume in the output properties
    outFileName = max(volume, key=volume.get)

    # Prepare metadata and commit
    properties = {
        "itk_volume": outFileName,
        "num_slices": n_slices,
        "segment_keys": {
            "radiology_patient_name": str(ds.PatientName),
            "radiology_accession_number": str(ds.AccessionNumber),
            "radiology_series_instance_uuid": str(ds.SeriesInstanceUID),
            "radiology_series_number": str(ds.SeriesNumber),
            "radiology_modality": str(ds.Modality),
        },
    }

    return properties


def check_pet(dcms):
    """Ensures dicom directory contains PET images

    Args:
        dcms (list[str]): list of dicom paths

    Returns:
        bool: True if all dicoms are PT modality
    """
    for dcm in dcms:
        ds = dcmread(dcm)
        if not ds.Modality == "PT":
            logger.warning(
                "check_pet - FAILED - Trying to apply PT corrections to non-PT image, gracefully skipping!"
            )
            return False

    logger.info("check_pet - PASSED - These are PT images")

    return True


def check_delay_correction(dcms):
    """Ensures all dicom images were delay corrected to their Aquisition TIme

    Args:
        dcms (list[str]): list of dicom paths
    """
    for dcm in dcms:
        ds = dcmread(dcm)
        if not ds.DecayCorrection == "START":
            logger.error(
                "check_delay_correction - FAILED - Cannot process a PET volume not constructed of 'START' time delay corrected slices!"
            )
            raise RuntimeError(
                "Cannot process a PET volume not constructed of 'START' time delay corrected slices!"
            )

    logger.info(
        "check_delay_correction - PASSED - All slices were decay corrected to their START AquisitionTime"
    )


def calculate_normalization(dcms):
    """Calculates the SUV normalization (g/BQ)

    Args:
        dcms (list[str]): list of dicom paths

    Returns:
        float: Normalization in (g/BQ)
    """
    logger.info("About to convert PET volume to SUV units")

    ds = dcmread(dcms[0])

    dose = float(ds.RadiopharmaceuticalInformationSequence[0].RadionuclideTotalDose)
    weight = float(ds.PatientWeight) * 1000

    logger.info(f"Radionuclide dose={dose}, patient weight={weight}")

    norm = weight / dose

    return norm


def convert_pet_volume_to_suv(dicom_urlpath, input_volume):
    """Renormalizes a PET ITK volume to SUVs, saves a new volume in-place

    Args:
        dicom_urlpath (str): path to matching dicom series within a folder
        input_volume (str): path to PT volume
    """
    dcms = list(Path(dicom_urlpath).glob("*.dcm"))

    if check_pet(dcms):
        check_delay_correction(dcms)

        norm = calculate_normalization(dcms)
        logger.info(f"Calculated SUV normalization factor={norm}")

        image, header = medpy.io.load(input_volume)

        image = image * norm
        logger.info(f"SUM SUV={image.sum()}")

        medpy.io.save(image, input_volume, hdr=header)

        logger.info(f"Saved normalized volume: {input_volume}")


def fire_cli():
    fire.Fire(dicom_to_itk)


if __name__ == "__main__":
    fire_cli()
