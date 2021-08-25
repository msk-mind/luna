# Uses numpy array containing annotation and splits into label masks. For each label mask, the annotations
# are vectorized into a set of polygons. These polygons are then converted into the geoJSON format and written to file.
from typing import List, Dict
from skimage import measure
import numpy as np
import pandas as pd
import json
import ast
import copy
import signal
import shapely 
from shapely.geometry import Polygon, MultiPolygon, shape, mapping

from dask.distributed import secede, rejoin

# max amount of time for a geojson to be generated. if generation surpasses this limit, it is likely the annotation file is
# too large or they may be annotation artifacts present in the slide. currently set at 30 minute timeout
TIMEOUT_SECONDS = 1800

DEFAULT_LABELSET_NAME = 'DEFAULT_LABELS'
# Base template for geoJSON file
geojson_base = {
    "type": "FeatureCollection",
    "features": []
}

def build_geojson_from_pointclick_json(labelsets: dict, labelset:str,
        sv_json:List[dict])-> list:
    """Build geoJSON m slideviewer JSON

    This method extracts point annotations from a slideviwer json object and 
    converts them to a standardized geoJSON format 

    Args:
        labelsets (dict): dictionary of label set as string (e.g. {labelset:
            {label_number: label_name}})
        labelset (str): the name of the labelset e.g. default_labels
        sv_json (list[dict]): annotatations from slideviwer in the form of a list of dictionaries 

    Returns:
        list: a list of geoJSON annotation objects
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


def find_parents(polygons: list) -> list:
    """determines of parent child relationships of polygons 

    Returns a list of size n (where n is the number of input polygons in the input list
    polygons) where the value at index n cooresponds to the nth polygon's parent. In
    the case of no parent, -1 is used. for example, parent_nums[0] = 2 means that
    polygon 0's parent is polygon 2

    Args:
        polygons (list): a list of shapely polygon objects
    
    Returns:
        list: a list of parent-child relationships for the polygon objects    

    """
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

def add_contours_for_label(annotation_geojson:Dict[str, any], annotation:np.ndarray,
        label_num:int, mappings:dict, contour_level:float) -> Dict[str, any]:
    """creates geoJSON feature dictionary for labels 

    Finds the contours for a label mask, builds a polygon and then converts the polygon 
    to geoJSON feature dictionary
    
    Args:
        annotation_geojson (dict[str, any]): geoJSON result to populate
        annotation (np.ndarray): npy array of bitmap
        label_num (int): the integer cooresponding to the annotated label
        mappings (dict): label map for specified label set
        contour_level (float): value along which to find contours in the array 

    Returns:
         dict[str, any]: geoJSON with label countours
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


def handler(signum:str, frame:str) -> None:
    """signal handler for geojson

    Args:
        signum (str): signal number
        fname (str): filename for which exception occurred

    Returns:
        None
    """

    raise TimeoutError("Geojson generation timed out.")


def build_labelset_specific_geojson(default_annotation_geojson:Dict[str, any],
        labelset:dict) -> Dict[str, any]:
    """builds geoJSON for labelset

    Instead of working with a large geJSON object, you can extact polygons
    that coorspond to specific labels into a smaller object. 

    Args:
        default_annotation_geojson (dict[str, any]):  geoJSON annotation file
        labelset (dict): label set dictionary   

    Returns:
        dict[str, any]: geoJSON with only polygons from provided labelset
    """

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


def build_all_geojsons_from_default(default_annotation_geojson:Dict[str, any],
        all_labelsets:List[dict], contour_level:float) -> dict:
    """builds geoJSON objects from a set of labels

    wraps build_labelset_specific_geojson with logic to generate annotations
    from multiple labelsets 

    Args:
        default_annotation_geojson (dict[str, any]): input geoJSON
        all_labelsets (list[dict]): a list of dictionaries containing label sets 
        contour_level (float):  value along which to find contours

    Returns:
        dict: a dictionary with labelset name and cooresponding geoJSON as key, value
        pairs
        
    """

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


def build_default_geojson_from_annotation(annotation_npy_filepath:str,
        all_labelsets:dict, contour_level:float):
    """builds geoJSONS from numpy annotation with default label set

    Args:
        annotation_npy_filepath (str): string to numpy annotation
        all_labelsets (dict): a dictionary of label sets
        contour_level (float):  value along which to find contours

    Returns:
        dict[str, any]: the default geoJSON annotation 
    """

    annotation = np.load(annotation_npy_filepath)
    default_annotation_geojson = copy.deepcopy(geojson_base)

    # signal logic doesn't work in dask distributed setup

    default_labelset = all_labelsets[DEFAULT_LABELSET_NAME]

    if not (annotation > 0).any():
        print(f"No annotated pixels detected in bitmap loaded from {annotation_npy_filepath}")
        return None

    # vectorize all
    for label_num in default_labelset:
        default_annotation_geojson = add_contours_for_label(default_annotation_geojson, annotation, label_num, default_labelset, float(contour_level))

    # empty geojson created, return nan and delete from geojson table
    if len(default_annotation_geojson['features']) == 0:
        print(f"Something went wrong with building default geojson from {annotation_npy_filepath}, quitting")
        return None

    return default_annotation_geojson


def build_geojson_from_annotation(df: pd.DataFrame) -> pd.DataFrame:
    """Builds geoJSON for all annotation labels in the specified labelset.

    Args:
        df (pandas.DataFrame): input regional annotation table 
    
    Returns:
        pandasDataFrame: dataframe with geoJSON field poopulated 
    """

    labelsets = df.label_config.values[0]
    annotation_npy_filepath = df.npy_filepath.values[0]
    labelset = df.labelset.values[0]
    contour_level = df.contour_level.values[0]

    labelsets = ast.literal_eval(labelsets)
    mappings = labelsets[labelset]

    print("\nBuilding GeoJSON annotation from npy file:", annotation_npy_filepath)

    annotation = np.load(annotation_npy_filepath)
    annotation_geojson = copy.deepcopy(geojson_base)

    signal.signal(signal.SIGALRM, handler)
    signal.alarm(TIMEOUT_SECONDS)

    try:
        for label_num in mappings:
            annotation_geojson = add_contours_for_label(annotation_geojson, annotation, label_num, mappings, float(contour_level))
    except TimeoutError as err:
        print("Timeout Error occured while building geojson from slide", annotation_npy_filepath)
        raise err

    # disables alarm
    signal.alarm(0)

    # empty geojson created, return nan and delete from geojson table
    if len(annotation_geojson['features']) == 0:
        return df

    df["geojson"] = json.dumps(annotation_geojson)
    return df


def concatenate_regional_geojsons(geojson_list: List[Dict[str, any]]) -> Dict[str, any]:
    """concatenate regional annotations

    Concatenates geojsons if there are more than one annotations for the labelset.

    Args: 
        geojson_list (list[dict[str, any]]): list of geoJSON strings

    Returns:
        dict[str, any]: a single concatenated geoJSON 
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
