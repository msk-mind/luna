# Uses numpy array containing annotation and splits into label masks. For each label mask, the annotations
# are vectorized into a set of polygons. These polygons are then converted into the geoJSON format and written to file.

from skimage import measure
import numpy as np
import json
import yaml, os
import copy
import signal

# max amount of time for a geojson to be generated. if generation surpasses this limit, it is likely the annotation file is
# too large or they may be annotation artifacts present in the slide. currently set at 30 minute timeout
TIMEOUT_SECONDS = 1800


# Base template for geoJSON file
geojson_base = {
    "type": "FeatureCollection",
    "features": []
}

# adapted from: https://github.com/ijmbarr/image-processing-with-numpy/blob/master/image-processing-with-numpy.ipynb
def add_contours_for_label(annotation_geojson, annotation, label_num, mappings, contour_level, polygon_tolerance):
    """
    Finds the contours for a label mask, builds a polygon, converts polygon to geoJSON feature dictionary

    :param annotation_geojson: geojson result to populate
    :param annotation: npy array of bitmap
    :param label_num: int value represented in the npy array; corresponding to the annotation label set.
    :param mappings: label map for the specified label set
    :param contour_level: value along which to find contours in the array
    :param polygon_tolerance: polygon resolution
    :return: geojson result
    """
    if label_num in annotation:
        print("Building contours for label " + str(label_num))

        num_pixels = np.count_nonzero(annotation == label_num)
        print("num_pixels with label", num_pixels)

        mask = np.where(annotation==label_num,1,0)
        contours = measure.find_contours(mask, level = contour_level)
        print("num contours", len(contours))

        scaled_tolerance = polygon_tolerance
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
def build_geojson_from_bitmap(configuration_file, dmt, annotation_npy_filepath, labelset, contour_level, polygon_tolerance):
    """
    Builds geojson for all annotation labels in the specified labelset.
    :param df: dataframe
    :return: dataframe populated with geojson_filepath, geojson_record_uuid
    """
    with open(configuration_file) as configfile:
        config = yaml.safe_load(configfile)

    config = config[dmt]
    mappings = config['label_sets'][labelset]

    print("\nBuilding GeoJSON annotation from npy file:", annotation_npy_filepath)

    annotation = np.load(annotation_npy_filepath)
    annotation_geojson = copy.deepcopy(geojson_base)

    signal.signal(signal.SIGALRM, handler)
    signal.alarm(TIMEOUT_SECONDS)

    try:
        for label_num in mappings:
            annotation_geojson = add_contours_for_label(annotation_geojson, annotation, label_num, mappings, float(contour_level), float(polygon_tolerance))
    except TimeoutError as err:
        print("Timeout Error occured while building geojson from slide", annotation_npy_filepath)
        df["geojson"] = np.nan
        df["geojson_record_uuid"] = np.nan

        return df

    # disables alarm
    signal.alarm(0)

    # empty geojson created, return nan and delete from geojson table
    if len(annotation_geojson['features']) == 0:
        return np.nan

    return annotation_geojson


def concatenate_geojsons_from_list(geojson_list):
    """
    Concatenate all geojsons in the list to one geojson.

    :param geojson_list: list of geojson file paths
    :return: extended feature map in json
    """
    base_geojson = geojson_list[0]

    if len(geojson_list) == 1:
        return base_geojson

    for json_dict in geojson_list[1:]:
        base_geojson['features'].extend(json_dict['features'])

    return base_geojson


def concatenate_regional_geojsons(geojson_list):
    """
    Concatenates geojsons wif there are more than one annotations for the labelset.
    :param df: dataframe
    :return: dataframe populated with concat_geojson_filepath, concat_geojson_record_uuid
    """
    geojson_list = [json.loads(geojson) for geojson in geojson_list]

    if len(geojson_list) == 1:
        return geojson_list[0]
    else:
        # create concatenated geojson
        concat_geojson = concatenate_geojsons_from_list(geojson_list)

    return concat_geojson
