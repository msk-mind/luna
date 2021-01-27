# Uses numpy array containing annotation and splits into label masks. For each label mask, the annotations
# are vectorized into a set of polygons. These polygons are then converted into the geoJSON format and written to file.

from skimage import measure
import numpy as np
import json
import yaml, os
import copy
import pandas as pd
import signal

from data_processing.common.utils import generate_uuid

# max amount of time for a geojson to be generated. if generation surpasses this limit, it is likely the annotation file is
# too large or they may be annotation artifacts present in the slide. currently set at 30 minute timeout
TIMEOUT_SECONDS = 1800


# Base template for geoJSON file
geojson_base = {
    "type": "FeatureCollection",
    "features": []
}

# finds the contours for a label mask, builds a polygon, converts polygon to geoJSON feature dictionary (annotation_geojson)
# adapted from: https://github.com/ijmbarr/image-processing-with-numpy/blob/master/image-processing-with-numpy.ipynb
def add_contours_for_label(annotation_geojson, annotation,label_num, mappings, CONTOUR_LEVEL, POLYGON_TOLERANCE):

    if label_num in annotation:
        print("Building contours for label " + str(label_num))

        num_pixels = np.count_nonzero(annotation == label_num)
        print("num_pixels with label", num_pixels)

        mask = np.where(annotation==label_num,1,0)
        contours = measure.find_contours(mask, level = CONTOUR_LEVEL)
        print("num contours", len(contours))

        scaled_tolerance = POLYGON_TOLERANCE
        if num_pixels >= 10000000:
            scaled_tolerance = int(num_pixels / 10000000)

        simplified_contours = [measure.approximate_polygon(c, tolerance=scaled_tolerance) for c in contours]

        for n, contour in enumerate(simplified_contours):
            feature_dict = {"type":"Feature", "properties":{}, "geometry":{"type":"Polygon", "coordinates": []}}
            feature_dict['properties']['label_num'] = label_num
            feature_dict['properties']['label_name'] = mappings[label_num]

            contour_list =   contour.tolist()
            for coord in contour_list:
                x = int(round(coord[0]))
                y = int(round(coord[1]))
                # switch coordinates, otherwise gets flipped
                coord[0] = y
                coord[1] = x

            feature_dict['geometry']['coordinates'] = contour_list
            annotation_geojson['features'].append(feature_dict)
    else:
        print("No label " + str(label_num) + " found")
    return annotation_geojson


def handler(signum, frame):
    raise TimeoutError("Geojson generation timed out.")


# pandas udf
def build_geojson_from_bitmap_pandas(df: pd.DataFrame) -> pd.DataFrame:

    CONFIGURATION_FILE = df.configuration_file.values[0]
    with open(CONFIGURATION_FILE) as configfile:
        config = yaml.safe_load(configfile)

    dmt = df.dmt.values[0]
    config = config[dmt]
    labelset = df.labelset.values[0]
    mappings = config['label_sets'][labelset]

    annotation_npy_filepath = df.npy_filepath.values[0]
    output_folder = df.slide_json_dir.values[0]

    CONTOUR_LEVEL = df.contour_level.values[0]
    POLYGON_TOLERANCE = df.polygon_tolerance.values[0]


    print("\nBuilding GeoJSON annotation from npy file:", annotation_npy_filepath)

    annotation = np.load(annotation_npy_filepath)
    annotation_geojson = copy.deepcopy(geojson_base)

    signal.signal(signal.SIGALRM, handler)
    signal.alarm(TIMEOUT_SECONDS)

    try:
        for label_num in mappings:
            annotation_geojson = add_contours_for_label(annotation_geojson, annotation, label_num, mappings, CONTOUR_LEVEL, POLYGON_TOLERANCE)
    except TimeoutError as err:
        print("Timeout Error occured while building geojson from slide", annotation_npy_filepath)
        df["geojson_filepath"] = np.nan
        df["geojson_record_uuid"] = np.nan

        return df

    # disables alarm
    signal.alarm(0)

    # empty geojson created, return nan and delete from geojson table
    if len(annotation_geojson['features']) == 0:
        df["geojson_filepath"] = np.nan
        df["geojson_record_uuid"] = np.nan
        return df

    new_image_name = os.path.basename(annotation_npy_filepath).replace(".npy","_geojson.json")
    caseid_folder = os.path.basename(os.path.dirname(annotation_npy_filepath))
    output_caseid_folder = output_folder + "/" + caseid_folder

    if not os.path.exists(output_caseid_folder):
        os.makedirs(output_caseid_folder)

    geojson_filepath = output_caseid_folder + "/" + new_image_name

    with open(geojson_filepath, "w") as geojson_out:
        json.dump(annotation_geojson, geojson_out)

    geojson_record_uuid = generate_uuid(geojson_filepath, ["SVGEOJSON", labelset])

    df["geojson_filepath"] = geojson_filepath
    df["geojson_record_uuid"] = geojson_record_uuid
    return df



def concatenate_geojsons_from_list(geojson_list):
    """
    Concatenate all geojsons in the list to one geojson.

    :param geojson_list: list of geojson file paths
    :return: extended feature map in json
    """
    if len(geojson_list) == 1:
        return geojson_list[0]
    else:
        base_geojson_filename = geojson_list[0]
        with open(base_geojson_filename) as base_geojson_file:
            base_geojson = json.load(base_geojson_file)

        for geojson_filename in geojson_list[1:]:
            with open(geojson_filename) as geojson_file:
                json_dict = json.load(geojson_file)
                base_geojson['features'].extend(json_dict['features'])

        print(base_geojson)
    return base_geojson


def concatenate_regional_geojsons_pandas(df: pd.DataFrame) -> pd.DataFrame:

    if len(df) == 1:
        df["concat_geojson_filepath"] = df.geojson_filepath.item()
        df["concat_geojson_record_uuid"] = df.geojson_record_uuid.item()
        return df
    else:
        labelset = df.labelset.values[0]
        output_folder = df.slide_json_dir.values[0]

        # create concatenated geojson
        geojson_list = list(df.geojson_filepath)
        concat_geojson = concatenate_geojsons_from_list(geojson_list)

        # same slide id across grouping
        slide_id = df.slide_id.values[0]
        new_image_name = slide_id+"_annot_concat_geojson.json"

        slideviewer_path = df.slideviewer_path.values[0]
        caseid_folder = slideviewer_path.replace(".svs", "")
        caseid_folder = caseid_folder.replace(";", "_")

        output_caseid_folder = os.path.join(output_folder, caseid_folder)

        if not os.path.exists(output_caseid_folder):
            os.makedirs(output_caseid_folder)

        geojson_filepath = os.path.join(output_caseid_folder, new_image_name)
        with open(geojson_filepath, "w") as geojson_out:
            json.dump(concat_geojson, geojson_out)

        concat_geojson_record_uuid = generate_uuid(geojson_filepath, ["SVCONCATGEOJSON", labelset])

        # set only first row so the rest can be purged from concat-table
        df.loc[0, 'concat_geojson_filepath'] = geojson_filepath
        df.loc[0, 'concat_geojson_record_uuid'] = concat_geojson_record_uuid
        df.loc[0, 'user'] = "CONCAT"
    return df