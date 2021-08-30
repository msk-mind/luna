import shutil
import click
import yaml, os, json
from datetime import datetime
from typing import Union, Tuple, List
import logging

from luna.common.CodeTimer import CodeTimer
from luna.common.config import ConfigSet
from luna.common.DataStore import DataStore_v2, DataStore

# from luna.common.sparksession import SparkConfig
import luna.common.constants as const
from luna.common.utils import get_absolute_path
from luna.pathology.common.utils import get_labelset_keys
import pandas as pd
import numpy as np

from luna.common.utils import get_absolute_path
from luna.pathology.common.slideviewer_client import fetch_slide_ids, download_zip, unzip
from luna.pathology.common.build_geojson \
    import build_default_geojson_from_annotation, build_all_geojsons_from_default, concatenate_regional_geojsons
import dask
from dask.distributed import Client, as_completed

from PIL import Image

import copy, shapely
from io import BytesIO
from skimage import measure

from filehash import FileHash

from datetime import datetime


os.environ['OPENBLAS_NUM_THREADS'] = '1'

# Base template for geoJSON file
geojson_base = {
    "type": "FeatureCollection",
    "features": []
}

# max amount of time for a geojson to be generated. if generation surpasses this limit, it is likely the annotation file is
# too large or they may be annotation artifacts present in the slide. currently set at 30 minute timeout
TIMEOUT_SECONDS = 1800

logger = logging.getLogger(__name__)

def get_slide_bitmap(full_filename:str, user:str, slide_id:str, SLIDE_BMP_DIR:str,
        SLIDEVIEWER_API_URL:str, TMP_ZIP_DIR:str, sv_project_id:str) -> Tuple[str, str]:
    """get slide bitmap

    Args:
        full_filename (str): filename of input slide
        user (str): name of pathologist/annotater who labled the input slide
        SLIDE_BMP_DIR (str): output folder to save bitmap to
        SLIDEVIEWER_API_URL (str): API url for slide viewer
        TMP_ZIP_DIR (str) temporary directory to save ziped bitmap files to
        sv_project_id (str): slide viewer project id

    Returns:
        Tuple[str, str]: a tuple of the bitmap record uuid and filepath to saved bitmap
    """

    full_filename_without_ext = full_filename.replace(".svs", "")

    bmp_dirname = os.path.join(SLIDE_BMP_DIR, full_filename_without_ext.replace(";", "_"))
    bmp_dest_path = os.path.join(bmp_dirname, str(slide_id) + '_' + user + '_annot.bmp')

    if os.path.exists(bmp_dest_path):
        logger.debug("Removing temporary file " + bmp_dest_path)
        os.remove(bmp_dest_path)

    # download bitmap file using api (from brush and fill tool), download zips into TMP_ZIP_DIR
    os.makedirs(TMP_ZIP_DIR, exist_ok=True)
    zipfile_path = os.path.join(TMP_ZIP_DIR, full_filename_without_ext + "_" + user + ".zip")

    url = SLIDEVIEWER_API_URL +'slides/'+ str(user) + '@mskcc.org/projects;' + str(sv_project_id) + ';' + full_filename + '/getLabelFileBMP'

    logger.debug(f"Pulling from Slideviewer URL={url}")

    success = download_zip(url, zipfile_path)

    bmp_record_uuid = 'n/a'
    bmp_filepath = 'n/a'

    if not success:
        os.remove(zipfile_path)
        return (bmp_record_uuid, bmp_filepath)

    unzipped_file_descriptor = unzip(zipfile_path)

    if unzipped_file_descriptor is None:
        return (bmp_record_uuid, bmp_filepath)

    # create bmp file from unzipped file
    os.makedirs(os.path.dirname(bmp_dest_path), exist_ok=True)
    with open(bmp_dest_path, "wb") as ff:
        ff.write(unzipped_file_descriptor.read("labels.bmp"))  # all bmps from slideviewer are called labels.bmp

    logger.info("Added slide " + str(slide_id) + " to " + str(bmp_dest_path) + "  * * * * ")

    bmp_hash = FileHash('sha256').hash_file(bmp_dest_path)
    bmp_record_uuid = f'SVBMP-{bmp_hash}'
    bmp_filepath = bmp_dirname + '/' + slide_id + '_' + user + '_' + bmp_record_uuid + '_annot.bmp'
    os.rename(bmp_dest_path, bmp_filepath)

    # cleanup
    if os.path.exists(zipfile_path):
        os.remove(zipfile_path)

    return (bmp_record_uuid, bmp_filepath)

def convert_bmp_to_npy(bmp_file:str, output_folder:str)->str:
    """convert bitmap to numpy

    Reads a bmp file and creates friendly numpy ndarray file in the uint8 format in the output
    directory specified, with extention .annot.npy

    Troubleshooting:
        Make sure Pillow is upgraded to version 8.0.0 if getting an Unsupported BMP Size OS Error

    Args:
        bmp_file (str): path to .bmp image
        output_folder (str): path to output folder

    Returns
        str: filepath to file containing numpy array
    """
    Image.MAX_IMAGE_PIXELS = None

    if not '.bmp' in bmp_file:
        return ''

    new_image_name = os.path.basename(bmp_file).replace(".bmp", ".npy")
    bmp_caseid_folder = os.path.basename(os.path.dirname(bmp_file))
    output_caseid_folder = os.path.join(output_folder, bmp_caseid_folder)

    if not os.path.exists(output_caseid_folder):
        os.makedirs(output_caseid_folder)

    output_filepath = os.path.join(output_caseid_folder, new_image_name)

    np.save(output_filepath, np.array(Image.open(bmp_file)))
    return output_filepath


def check_slideviewer_and_download_bmp(sv_project_id:str, slideviewer_path:str,
        slide_id:str, users:List, SLIDE_BMP_DIR:str, SLIDEVIEWER_API_URL:str,
        TMP_ZIP_DIR:str) -> Union[None, List]:
    """download bitmap annotation from slideviwer

    Args:
        sv_project_id (str): slideviewer project id
        slideviewer_path (str): filepath to the input slide
        slide_id (str): slide id
        users (List[str]): list of users who provided annotations
        SLIDE_BMP_DIR (str): output folder to save bitmap to
        SLIDEVIEWER_API_URL (str): API url for slide viewer
        TMP_ZIP_DIR (str) temporary directory to save ziped bitmap files to

    Returns:
        Union[None, List]: returns none if there are no annotations to process, or
            returns a list containing output parameters
    """
    slide_id = str(slide_id)

    outputs = []
    output_dict_base = {
        "sv_project_id":sv_project_id,
        "slideviewer_path": slideviewer_path,
        "slide_id": slide_id,
        "user": "n/a",
        "bmp_filepath": 'n/a',
        "npy_filepath": 'n/a',
        "geojson": 'n/a',
        "geojson_path": 'n/a',
        "date": datetime.now()
    }
    outputs.append(output_dict_base)

    for user in users:
        # download bitmap
        bmp_record_uuid, bmp_filepath = get_slide_bitmap(slideviewer_path, user, slide_id, SLIDE_BMP_DIR, SLIDEVIEWER_API_URL, TMP_ZIP_DIR, sv_project_id)
        # convert to npy
        if bmp_record_uuid != 'n/a' or bmp_filepath != 'n/a':

            output_dict = copy.deepcopy(output_dict_base)
            output_dict['user'] = user
            output_dict["bmp_filepath"] = bmp_filepath
            outputs.append(output_dict)
    # at this point if outputs is empty, return early
    if len(outputs) <= 1:
        return None
    else:
        return outputs


def convert_slide_bitmap_to_geojson(outputs, all_labelsets:List[dict],
        contour_level:float, SLIDE_NPY_DIR:str, slide_store_dir:str) -> Tuple[str, List]:
    """convert slide bitmap to geoJSON

    Args:
        outputs (List[dict]): list of output parameter dict
        all_labelsets (List[dict]): a list of dictionaries containing label sets
        contour_level (float): value along which to find contours
        SLIDE_NPY_DIR (str): directory containing the slide saved as a .npy
        slide_store_dir (str): directory of the datastore

    Returns:
        Tuple[str, List]: a pair of slide id and output geojson tables
    """
    outputs = copy.deepcopy(outputs)
    try:
        slide_id = outputs[0]['slide_id']
        geojson_table_outs = []
        concat_geojson_table_outs = []
        output_dict_base = outputs.pop(0)

        logger.info(f" >>>>>>> Processing [{slide_id}] <<<<<<<<")

        store = DataStore_v2(slide_store_dir)

        for user_annotation in outputs:
            bmp_filepath = user_annotation['bmp_filepath']
            npy_filepath = convert_bmp_to_npy(bmp_filepath, SLIDE_NPY_DIR)
            user_annotation['npy_filepath'] = npy_filepath

            store.put (npy_filepath, store_id=user_annotation['slide_id'], namespace_id=user_annotation['user'], data_type='RegionalAnnotationBitmap')


        # build geojsons
        for user_annotation in outputs:
            npy_filepath = user_annotation['npy_filepath']
            default_annotation_geojson = build_default_geojson_from_annotation(npy_filepath, all_labelsets, contour_level)

            if not default_annotation_geojson:
                raise RuntimeError("Error while building default geojson!!!")

            user_annotation['geojson'] = default_annotation_geojson


        for user_annotation in outputs:
            default_annotation_geojson = user_annotation['geojson']
            labelset_name_to_labelset_specific_geojson = build_all_geojsons_from_default(default_annotation_geojson, all_labelsets, contour_level)
            for labelset_name, geojson in labelset_name_to_labelset_specific_geojson.items():
                geojson_table_out_entry = copy.deepcopy(user_annotation)
                geojson_table_out_entry['labelset'] = labelset_name
                geojson_table_out_entry['geojson'] = geojson

                path = store.write (json.dumps(geojson, indent=4), store_id=user_annotation['slide_id'], namespace_id=user_annotation['user'], data_type='RegionalAnnotationJSON', data_tag=labelset_name)
                geojson_table_out_entry['geojson_path'] = path

                geojson_table_outs.append(geojson_table_out_entry)



        geojsons_to_concat = [json.dumps(user_annotation['geojson']) for user_annotation in outputs]
        concat_default_annotation_geojson = concatenate_regional_geojsons(geojsons_to_concat)
        labelset_name_to_labelset_specific_geojson = build_all_geojsons_from_default(concat_default_annotation_geojson, all_labelsets, contour_level)
        for labelset_name, geojson in labelset_name_to_labelset_specific_geojson.items():
            concat_geojson_table_out_entry = copy.deepcopy(output_dict_base)
            concat_geojson_table_out_entry['user'] = "CONCAT"
            concat_geojson_table_out_entry['labelset'] = labelset_name
            concat_geojson_table_out_entry['geojson'] = geojson

            path = store.write(json.dumps(geojson, indent=4), store_id=concat_geojson_table_out_entry['slide_id'], namespace_id=concat_geojson_table_out_entry['user'], data_type='RegionalAnnotationJSON', data_tag=labelset_name)

            concat_geojson_table_out_entry['geojson_path'] = path

            concat_geojson_table_outs.append(concat_geojson_table_out_entry)

        return slide_id, geojson_table_outs + concat_geojson_table_outs

    except Exception as exc:
        logger.exception(f"{exc}, stopping job execution on {slide_id}...", extra={'slide_id':slide_id})
        raise exc
