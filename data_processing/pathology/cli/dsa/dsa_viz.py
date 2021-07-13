import click
from decimal import Decimal
import pandas as pd
import json, geojson, ijson
import copy
import time, os
from PIL import Image
import re
import numpy as np

from data_processing.pathology.cli.dsa.dsa_api_handler import get_item_uuid, push_annotation_to_dsa_image, system_check
from data_processing.pathology.cli.dsa.utils import get_color, get_continuous_color, \
    vectorize_np_array_bitmask_by_pixel_value

# Base DSA jsons
base_dsa_polygon_element = {"fillColor": "rgba(0, 0, 0, 0)", "lineColor": "rgb(0, 0, 0)","lineWidth": 2,"type": "polyline","closed": True, "points": [], "label": {"value": ""}}
base_dsa_point_element = {"fillColor": "rgba(0, 0, 0, 0)", "lineColor": "rgb(0, 0, 0)","lineWidth": 2,"type": "point", "center": [], "label": {"value": ""}}
base_dsa_annotation  = {"description": "", "elements": [], "name": ""}

# Qupath 20x mag factor
QUPATH_MAG_FACTOR = 0.5011
image_id_regex = "(.*).svs"

# accepts list of filepaths to check if path exists
def check_filepaths_valid(filepaths):

    all_files_found = True
    for filepath in filepaths:
        if not os.path.exists(filepath):
            print("ERROR: filepath in config: ", filepath, 'does not exist')
            all_files_found = False 
    return all_files_found

def save_dsa_annotation(base_annotation, elements, annotation_name, output_folder, image_filename) :
    """ 
    Populate base annotations, save to json outfile, and push to DSA
    """
    dsa_annotation = copy.deepcopy(base_annotation)

    dsa_annotation["elements"] = elements

    dsa_annotation["name"] = annotation_name

    image_id = re.search(image_id_regex, image_filename).group(1)
    annotation_name_replaced = annotation_name.replace(" ","_")

    os.makedirs(output_folder, exist_ok=True)
    outfile_name = os.path.join(output_folder, f"{annotation_name_replaced}_{image_id}.json")

    try:
        with open(outfile_name, 'w') as outfile:
            json.dump(dsa_annotation, outfile)
        return outfile_name
    except Exception as e:
        print("ERROR: write permissions needs to be enabled for: ", os.path.dirname(outfile_name))
        return None


@click.command()
@click.option("-d", "--data_config",
              help="path to data config file",
              required=True,
              type=click.Path(exists=True))
@click.option("-s", "--source_type",
              help="string describing data source that is to be transformed into a dsa json. \
               supports stardist-polygon, stardist-cell, regional-polygon, qupath-polygon, bitmask-polygon, heatmap",
              required=True)
def cli(data_config, source_type):
    """
    DSA annotation builder
    """

    functions = {"stardist-polygon": stardist_polygon, 
                "stardist-cell": stardist_cell, 
                "regional-polygon": regional_polygon, 
                "qupath-polygon": qupath_polygon, 
                "bitmask-polygon": bitmask_polygon,
                "heatmap": heatmap}
    
    if source_type not in functions:
            print(f"ERROR: source_type {source_type} is not supported")
    else:   
        annotation_filepath = functions[source_type](data_config)
        print(f"Annotation written to {annotation_filepath}")


def stardist_polygon(data_config):
    """
    Upload stardist geojson classification results
    """
    with open(data_config) as config_json:
        data = json.load(config_json)

    if not check_filepaths_valid([data['input']]):
        return 

    print("Building annotation for image: {}".format(data['image_filename']))
    start = time.time()

    # can't handle NaNs for vectors, do this to replace all NaNs
    # TODO: find better fix
    # for now: https://stackoverflow.com/questions/17140886/how-to-search-and-replace-text-in-a-file

    with open(data["input"], 'r') as input_file:
        filedata = input_file.read()
    newdata = filedata.replace("NaN","-1")

    elements = []
    for cell in ijson.items(newdata, "item"):
        label_name = cell['properties']['classification']['name']
        coord_list = list(cell['geometry']['coordinates'][0])

        # uneven nested list when iterative parsing of json --> make sure to get the list of coords
        # this can come as mixed types as well, so type checking needed
        while isinstance(coord_list, list) and isinstance(coord_list[0], list) and not isinstance(coord_list[0][0], (int,float,Decimal)):
            coord_list = coord_list[0]

        coords = [ [float(coord[0]), float(coord[1]), 0] for coord in coord_list]
        element = copy.deepcopy(base_dsa_polygon_element)

        element["label"]["value"] = label_name
        line_color, fill_color = get_color(label_name, data["line_colors"], data["fill_colors"])
        element["fillColor"] = fill_color
        element["lineColor"] = line_color
        element["points"] = coords

        elements.append(element)

    print("Time to build annotation", time.time() - start)

    annotatation_filepath = save_dsa_annotation(base_dsa_annotation, elements, data["annotation_name"], data["output_folder"], data["image_filename"])
    return annotatation_filepath

def stardist_cell(data_config):
    """
    Upload TSV classification data
    """
    with open(data_config) as config_json:
        data = json.load(config_json)


    if not check_filepaths_valid([data['input']]):
        return 


    print("Building annotation for image: {}".format(data['image_filename']))
    start = time.time()

    # qupath_stardist_cell_tsv can be quite large to load all columns into memory (contains many feature columns), so only load baisc columns that are needed for now
    cols_to_load = ["Name", "Class", "ROI", "Centroid X µm", "Centroid Y µm", "Parent"]
    df = pd.read_csv(data["input"], sep ="\t", usecols=cols_to_load)

    # do some preprocessing on the tsv -- e.g. stardist sometimes finds cells in glass
    df = df[df["Parent"] != "Glass"]

    # populate json elements
    elements = []
    for idx, row in df.iterrows():

        elements_entry = copy.deepcopy(base_dsa_point_element)

        # x,y coordinates from stardist are in microns so divide by QUPATH_MAG_FACTOR = 0.5011 (exact 20x mag factor used by qupath specifically)
        x = row["Centroid X µm"]/QUPATH_MAG_FACTOR
        y = row["Centroid Y µm"]/QUPATH_MAG_FACTOR

        # Get cell label and add to element
        label_name = row["Class"]
        elements_entry["label"]["value"] = label_name

        # get color and add to element
        line_color, fill_color = get_color(label_name, data["line_colors"], data["fill_colors"])
        elements_entry["fillColor"] = fill_color
        elements_entry["lineColor"] = line_color

        # add centroid coordinate of cell to element
        center = [x,y,0]
        elements_entry["center"] = center

        elements.append(elements_entry)

    print("Time to build annotation", time.time() - start)

    annotatation_filepath = save_dsa_annotation(base_dsa_annotation, elements, data["annotation_name"], data["output_folder"], data["image_filename"])
    return annotatation_filepath

def regional_polygon(data_config):
    """
    Upload regional annotation data
    """
    with open(data_config) as config_json:
        data = json.load(config_json)


    if not check_filepaths_valid([data['input']]):
        return 

    print("Building annotation for image: {}".format(data['image_filename']))
    start = time.time()

    with open(data["input"]) as regional_file:
        regional_annotation = geojson.load(regional_file)

    elements = []
    for annot in regional_annotation['features']:

        # get label name and add to element
        element = copy.deepcopy(base_dsa_polygon_element)
        label_name = annot.properties['label_name']
        element["label"]["value"] = label_name

        # get label specific color and add to element
        line_color, fill_color = get_color(label_name, data["line_colors"], data["fill_colors"])
        element["fillColor"] = fill_color
        element["lineColor"] = line_color

        # add coordinates
        coords = annot['geometry']['coordinates']
        for c in coords:
            c.append(0)
        element["points"] = coords
        elements.append(element)

    print("Time to build annotation", time.time() - start)

    annotatation_filepath = save_dsa_annotation(base_dsa_annotation, elements, data["annotation_name"], data["output_folder"], data["image_filename"])
    return annotatation_filepath

def qupath_polygon(data_config):
    """
    Upload regional annotation data
    """
    with open(data_config) as config_json:
        data = json.load(config_json)

    if not check_filepaths_valid([data['input']]):
        return 

    print("Building annotation for image: {}".format(data['image_filename']))
    start = time.time()

    with open(data["input"]) as regional_file:
        pixel_clf_polygons = geojson.load(regional_file)

    elements = []
    for polygon in pixel_clf_polygons:

        props = polygon.properties
        if 'classification' not in props:
            continue

        label_name = polygon.properties['classification']['name']
        if label_name in data['classes_to_include']:

            element = copy.deepcopy(base_dsa_polygon_element)
            element["label"]["value"] = label_name
            # get label specific color and add to element
            line_color, fill_color = get_color(label_name, data["line_colors"], data["fill_colors"])
            element["fillColor"] = fill_color
            element["lineColor"] = line_color
            
            coords = polygon['geometry']['coordinates']

            # uneven nesting of connected components
            for coord in coords:
                if isinstance(coord[0], list) and isinstance(coord[0][0], (int,float)):
                    for c in coord:     
                        c.append(0)
                    element["points"] = coord
                    elements.append(element)
                else:
                    for i in range(len(coord)):
                        connected_component_coords = coord[i]
                        connected_component_element = copy.deepcopy(element)
                        for c in connected_component_coords:
                            c.append(0)

                        connected_component_element["points"] = connected_component_coords
                        elements.append(connected_component_element)

    print("Time to build annotation", time.time() - start)

    annotatation_filepath = save_dsa_annotation(base_dsa_annotation, elements, data["annotation_name"], data["output_folder"], data["image_filename"])
    return annotatation_filepath


def bitmask_polygon(data_config):
    """
    Upload bitmask PNGs
    """
    with open(data_config) as config_json:
        data = json.load(config_json)

    bitmask_filepaths = list(data['input'].values())
    if not check_filepaths_valid(bitmask_filepaths):
        return 

    print("Building annotation for image: {}".format(data['image_filename']))
    start = time.time()

    elements = []
    for bitmask_label, bitmask_filepath in data["input"].items():
        Image.MAX_IMAGE_PIXELS = 5000000000
        annotation = Image.open(bitmask_filepath)
        bitmask_np = np.array(annotation)
        simplified_contours = vectorize_np_array_bitmask_by_pixel_value(bitmask_np)

        for n, contour in enumerate(simplified_contours):
            element = copy.deepcopy(base_dsa_polygon_element)
            label_name = bitmask_label
            element["label"]["value"] = label_name

            # get label specific color and add to element
            line_color, fill_color = get_color(label_name, data["line_colors"], data["fill_colors"])
            element["fillColor"] = fill_color
            element["lineColor"] = line_color

            coords = contour.tolist()
            for c in coords:
                c.append(0)
            element["points"] = coords
            elements.append(element)

    print("Time to build annotation", time.time() - start)

    annotatation_filepath = save_dsa_annotation(base_dsa_annotation, elements, data["annotation_name"], data["output_folder"], data["image_filename"])
    return annotatation_filepath

def heatmap(data_config):
    """
    Upload heatmap based on tile scores
    """
    with open(data_config) as config_json:
        data = json.load(config_json)

    if not check_filepaths_valid([data['input']]):
        return 

    print("Building annotation for image: {}".format(data['image_filename']))
    start = time.time()

    df = pd.read_csv(data["input"])
    scaled_tile_size = int(data["tile_size"]) * int(data["scale_factor"])

    elements = []
    for _, row in df.iterrows():
        element = copy.deepcopy(base_dsa_polygon_element)
        label_value = row[data["column"]]
        element["label"]["value"] = str(label_value)

        # get label specific color and add to elements
        line_color, fill_color = get_continuous_color(label_value)
        element["fillColor"] = fill_color
        element["lineColor"] = line_color

        # convert coordinate string to tuple using eval
        x,y = eval(row["coordinates"])

        pixel_x = x * scaled_tile_size
        pixel_y = y * scaled_tile_size

        coords = [[pixel_x, pixel_y], [pixel_x+scaled_tile_size, pixel_y], [pixel_x+scaled_tile_size, pixel_y+scaled_tile_size], [pixel_x, pixel_y+scaled_tile_size],[pixel_x, pixel_y]]
        for c in coords:
            c.append(0)
        element["points"] = coords
        elements.append(element)

    annotation_name = data["column"] + "_" + data["annotation_name"]

    annotatation_filepath = save_dsa_annotation(base_dsa_annotation, elements, annotation_name, data["output_folder"], data["image_filename"])
    return annotatation_filepath


if __name__ == '__main__':
    cli()
