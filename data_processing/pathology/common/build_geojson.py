# Uses numpy array containing annotation and splits into label masks. For each label mask, the annotations
# are vectorized into a set of polygons. These polygons are then converted into the geoJSON format and written to file.

from skimage import measure
import numpy as np
import json
import ast
import copy
import signal
import shapely 
from shapely.geometry import Polygon, MultiPolygon, shape, mapping

# max amount of time for a geojson to be generated. if generation surpasses this limit, it is likely the annotation file is
# too large or they may be annotation artifacts present in the slide. currently set at 30 minute timeout
TIMEOUT_SECONDS = 1800

DEFAULT_LABELSET_NAME = 'DEFAULT_LABELS'
# Base template for geoJSON file
geojson_base = {
    "type": "FeatureCollection",
    "features": []
}

def build_geojson_from_pointclick_json(labelsets, labelset, sv_json):
    """
    Build geojson from slideviewer json.

    :param labelsets: dictionary of labelset as string {labelset: {label number: label name}}
    :param labelset: a labelset e.g. default_labels
    :param sv_json: list of dictionaries, from slideviewer
    :return: geojson list
    """

    labelsets = ast.literal_eval(labelsets)
    mappings = labelsets[labelset]

    output_geojson = []
    for entry in sv_json:
        point = {}
        x = int(entry['x'])
        y = int(entry['y'])
        class_num = int(entry['class'])
        if class_num not in mappings:
            continue
        class_name = mappings[class_num]
        coordinates = [x,y]

        point["type"] = "Feature"
        point["id"] = "PathAnnotationObject"
        point["geometry"] = {"type": "Point",  "coordinates": coordinates}
        point["properties"] = {"classification": {"name": class_name}}
        output_geojson.append(point)

    return output_geojson


# determines parent child relationships of polygons
# returns a list of size n (where n is the number of input polygons in input list polygons)
# each index in n corresponds to polygon n's parent. in case of no parent -1 is used.
# for example, parent_nums[0] == 2 means that polygon 0's parent is polygon 2.
def find_parents(polygons):
    parent_nums = []
    for child in polygons:
        found_parent = False
        for parent_idx, parent in enumerate(polygons):
            if child == parent:
                continue
            # found parent for child
            if parent.contains(child):
                parent_nums.append(parent_idx)
                found_parent = True
                break
        # finished looping through all potential parents, so child is a parent
        if not found_parent:
            parent_nums.append(-1)
    print(parent_nums)
    return parent_nums

# TODO test performance with/without polygon-tolerance. approximate_polygons(polygon_tolerance) might just be a slow and unnecessary step. 
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

        mask = np.where(annotation==label_num,1,0).astype(np.int8)
        contours = measure.find_contours(mask, level = contour_level)
        print("num_contours", len(contours))

        polygons = [Polygon(np.squeeze(c)) for c in contours]
        parent_nums = find_parents(polygons)

        polygon_by_index_number = {}

        for index, parent in enumerate(parent_nums):
            contour = contours[index]
            contour_list = contour.tolist()
            
            # switch coordinates, otherwise gets flipped
            for coord in contour_list:
                x = int(coord[0])
                y = int(coord[1])
                coord[0] = y
                coord[1] = x
            
            # this polygon does not have parent, so this is a parent object (top level)
            if parent == -1:
                polygon = {"type":"Feature", "properties":{}, "geometry":{"type":"Polygon", "coordinates": []}}
                polygon['properties']['label_num'] = int(label_num)
                polygon['properties']['label_name'] = mappings[label_num]
                polygon['geometry']['coordinates'].append(contour_list)
                polygon_by_index_number[index] = polygon
            else:
                # this is a child object, add coordinates as a hole to the parent polygon
                
                # fetch parent's polygon 
                parent_polygon = polygon_by_index_number[parent]

                # append as hole to parent
                parent_polygon['geometry']['coordinates'].append(contour_list)

        # add parent polygon feature dicts to running annotation geojson object
        for index, polygon in polygon_by_index_number.items():
            annotation_geojson['features'].append(polygon)

    else:
        print("No label " + str(label_num) + " found")
    return annotation_geojson


def handler(signum, frame):
    raise TimeoutError("Geojson generation timed out.")


def build_labelset_specific_geojson(default_annotation_geojson, labelset):

    annotation_geojson = copy.deepcopy(geojson_base)

    for feature in default_annotation_geojson['features']:

        # number is fixed
        label_num = feature['properties']['label_num']
        # add polygon to json, change name potentially needed
        if label_num in labelset:
            new_feature_polygon = copy.deepcopy(feature)

            # get new name and change
            new_label_name = labelset[label_num]
            new_feature_polygon['properties']['label_name'] = new_label_name

            # add to annotation_geojson being built
            annotation_geojson['features'].append(new_feature_polygon)

    # no polygons containing labels in labelset
    if len(annotation_geojson['features']) == 0:
        return None

    return annotation_geojson


def build_all_geojsons_from_default(default_annotation_geojson, all_labelsets, contour_level, polygon_tolerance):

    labelset_name_to_labelset_specific_geojson = {}
    
    for labelset_name, labelset in all_labelsets.items():
        if labelset_name != DEFAULT_LABELSET_NAME:
            # use default labelset geojson to build labelset specific geojson
            annotation_geojson = build_labelset_specific_geojson(default_annotation_geojson, labelset)
        else:
            annotation_geojson = default_annotation_geojson

        # only add if geojson not none (built correctly and contains >= 1 polygon)
        if annotation_geojson:
            labelset_name_to_labelset_specific_geojson[labelset_name] = json.dumps(annotation_geojson)
        
    return labelset_name_to_labelset_specific_geojson


def build_default_geojson_from_annotation(annotation_npy_filepath, all_labelsets, contour_level, polygon_tolerance):

    annotation = np.load(annotation_npy_filepath)
    default_annotation_geojson = copy.deepcopy(geojson_base)
    # signal logic doesn't work in dask distributed setup

    default_labelset = all_labelsets[DEFAULT_LABELSET_NAME]

    if not (annotation > 0).any():
        raise ValueError(f"No annotated pixels detected in bitmap loaded from {annotation_npy_filepath}")

    # vectorize all
    for label_num in default_labelset:
        default_annotation_geojson = add_contours_for_label(default_annotation_geojson, annotation, label_num, default_labelset, float(contour_level), float(polygon_tolerance))

    # empty geojson created, return nan and delete from geojson table
    if len(default_annotation_geojson['features']) == 0:
        raise RuntimeError(f"Something went wrong with building default geojson from {annotation_npy_filepath}, quitting")

    return default_annotation_geojson

def build_geojson_from_annotation(df):
    """
    Builds geojson for all annotation labels in the specified labelset.

    :param df: Pandas dataframe
    :return: Pandas dataframe with geojson field populated
    """

    labelsets = df.label_config.values[0]
    annotation_npy_filepath = df.npy_filepath.values[0]
    labelset = df.labelset.values[0]
    contour_level = df.contour_level.values[0]
    polygon_tolerance = df.polygon_tolerance.values[0]

    labelsets = ast.literal_eval(labelsets)
    mappings = labelsets[labelset]

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
        raise

    # disables alarm
    signal.alarm(0)

    # empty geojson created, return nan and delete from geojson table
    if len(annotation_geojson['features']) == 0:
        return df

    df["geojson"] = json.dumps(annotation_geojson)
    return df


def concatenate_regional_geojsons(geojson_list):
    """
    Concatenates geojsons if there are more than one annotations for the labelset.

    :param geojson_list: list of geojson strings
    :return: concatenated geojson dict
    """
    # create json from str representations
    geojson_list = [json.loads(geojson) for geojson in geojson_list]

    concat_geojson = geojson_list[0]
    if len(geojson_list) == 1:
        return concat_geojson

    # create concatenated geojson
    for json_dict in geojson_list[1:]:
        print(f"Concatenating {len(geojson_list)} geojsons")
        concat_geojson['features'].extend(json_dict['features'])

    return concat_geojson
