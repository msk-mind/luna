# Uses numpy array containing annotation and splits into label masks. For each label mask, the annotations
# are vectorized into a set of polygons. These polygons are then converted into the geoJSON format and written to file.

from skimage import measure
import numpy as np
import json
import ast
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

def build_geojson_from_pointclick_json(labelsets, labelset, sv_json):
    """
    Build geojson from slideviewer json.

    :param labelsets: dictionary of labelset as string {labelset: {label number: label name}}
    :param labelset: a labelset e.g. default_labels
    :param sv_json: list of dictionaries, from slideviewer
    :return: geojson list
    """
    print("Building geojson for labelset " + str(labelset))

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
        print("num contours", len(contours))

        scaled_tolerance = polygon_tolerance
        if num_pixels >= 10000000:
            scaled_tolerance = int(num_pixels / 10000000)

        simplified_contours = [measure.approximate_polygon(c, tolerance=scaled_tolerance) for c in contours]

        for n, contour in enumerate(simplified_contours):
            feature_dict = {"type":"Feature", "properties":{}, "geometry":{"type":"Polygon", "coordinates": []}}
            feature_dict['properties']['label_num'] = str(label_num)
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


#def build_geojson_from_annotation(labelsets, annotation_npy_filepath, labelset, contour_level, polygon_tolerance):
def build_geojson_from_annotation(df):
    """
    Builds geojson for all annotation labels in the specified labelset.

    :param labelsets: dictionary of labelset as string {labelset: {label number: label name}}
    :param annotation_npy_filepath: path to annotation npy file
    :param labelset: a labelset e.g. default_labels
    :param contour_level: value along which to find contours in the array
    :param polygon_tolerance: polygon resolution
    :return:
    """
    from build_geojson import add_contours_for_label, handler

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

        return None

    # disables alarm
    signal.alarm(0)

    # empty geojson created, return nan and delete from geojson table
    if len(annotation_geojson['features']) == 0:
        return None

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

