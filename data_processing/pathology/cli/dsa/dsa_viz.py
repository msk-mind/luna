import click
import pandas as pd
import json, geojson, ijson
import copy
import time, os
from PIL import Image
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

def save_push_results(base_annotation, elements, annotation_name, image_name, uri, token):
    """
    Populate base annotations, save to json outfile, and push to DSA
    """
    dsa_annotation = copy.deepcopy(base_annotation)

    dsa_annotation["elements"] = elements
    dsa_annotation["name"] = annotation_name

    outfile_name = annotation_name.replace(" ","_") + ".json"
    with open(outfile_name, 'w') as outfile:
        json.dump(dsa_annotation, outfile)

    dsa_uuid = get_item_uuid(image_name, uri, token)
    push_annotation_to_dsa_image(dsa_uuid, dsa_annotation, uri, token)


@click.group()
@click.pass_context
@click.option('-c', '--config',
              help="json including DSA host, port and token info",
              type=click.Path(exists=True),
              required=True)
def cli(ctx, config):
    """
    DSA visualization CLI
    """
    ctx.ensure_object(dict)

    with open(config) as config_json:
        config_data = json.load(config_json)

    # Girder Token can be found in the DSA API Swagger Docs under 'token': (http://{host}:8080/api/v1#!/token/token_currentSession)
    ctx.obj["uri"] = config_data["host"] + ":" + config_data["port"]
    ctx.obj["token"] = config_data["token"]
    # print(ctx.obj)

    # TODO use girder client
    # https://girder.readthedocs.io/en/latest/python-client.html#the-python-client-library

    # check DSA connection
    system_check(ctx.obj["uri"], ctx.obj["token"])


@cli.command()
@click.pass_context
@click.option("-d", "--data_config",
              help="path to data config file",
              required=True,
              type=click.Path(exists=True))
def stardist_polygon(ctx, data_config):
    """
    Upload stardist geojson classification results
    """
    with open(data_config) as config_json:
        data = json.load(config_json)

    print("Starting upload for image: {}".format(data['image_name']))
    start = time.time()

    # can't handle NaNs for vectors, do this to replace all NaNs
    # TODO: find better fix
    # for now: https://stackoverflow.com/questions/17140886/how-to-search-and-replace-text-in-a-file
    new_filepath = data["input"].replace(".geojson", "_NAN_modified.geojson")
    if not os.path.exists(new_filepath):
        with open(data["input"], 'r') as input_file:
            filedata = input_file.read()

        newdata = filedata.replace("NaN","-1")
        with open(new_filepath,'w') as new_file:
            new_file.write(newdata)

    new_file = open(new_filepath, 'r')
    elements = []
    for cell in ijson.items(new_file, "item"):
        label_name = cell['properties']['classification']['name']
        coord_list = list(cell['geometry']['coordinates'][0])

        # uneven nested list when iterative parsing of json --> make sure to get the list of coords
        while isinstance(coord_list, list) and len(coord_list) <= 1:
            #print("invoked")
            coord_list = coord_list[0]

        coords = [ [float(coord[0]), float(coord[1]), 0] for coord in coord_list]
        element = copy.deepcopy(base_dsa_polygon_element)

        element["label"]["value"] = label_name
        line_color, fill_color = get_color(label_name, data["line_colors"], data["fill_colors"])
        element["fillColor"] = fill_color
        element["lineColor"] = line_color
        element["points"] = coords

        elements.append(element)

    new_file.close()
    print("Time to build annotation", time.time() - start)
    print(len(elements)) # 981070

    save_push_results(base_dsa_annotation, elements, data["output"], data["image_name"],
                      ctx.obj['uri'], ctx.obj['token'])


@cli.command()
@click.pass_context
@click.option("-d", "--data_config",
              help="path to data config file",
              required=True,
              type=click.Path(exists=True))
def stardist_cell(ctx, data_config):
    """
    Upload TSV classification data
    """
    with open(data_config) as config_json:
        data = json.load(config_json)

    print("Starting upload for image: {}".format(data['image_name']))
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

    save_push_results(base_dsa_annotation, elements, data["output"], data["image_name"],
                      ctx.obj['uri'], ctx.obj['token'])


@cli.command()
@click.pass_context
@click.option("-d", "--data_config",
              help="path to data config file",
              required=True,
              type=click.Path(exists=True))
def regional_polygon(ctx, data_config):
    """
    Upload regional annotation data
    """
    with open(data_config) as config_json:
        data = json.load(config_json)

    print("Starting upload for image: {}".format(data['image_name']))
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

    save_push_results(base_dsa_annotation, elements, data["output"], data["image_name"],
                      ctx.obj['uri'], ctx.obj['token'])


@cli.command()
@click.pass_context
@click.option("-d", "--data_config",
              help="path to tile level csv",
              type=click.Path(exists=True),
              required=True)
def bitmask_polygon(ctx, data_config):
    """
    Upload bitmask PNGs
    """
    with open(data_config) as config_json:
        data = json.load(config_json)

    print("Starting upload for image: {}".format(data['image_name']))
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

    save_push_results(base_dsa_annotation, elements, data["output"], data["image_name"],
                      ctx.obj['uri'], ctx.obj['token'])


@cli.command()
@click.pass_context
@click.option("-d", "--data_config",
              help="path to tile level csv",
              type=click.Path(exists=True),
              required=True)
def heatmap(ctx, data_config):
    """
    Upload heatmap based on tile scores
    """
    with open(data_config) as config_json:
        data = json.load(config_json)

    print("Starting upload for image: {}".format(data['image_name']))
    start = time.time()

    df = pd.read_csv(data["input"])
    scaled_tile_size = int(data["tile_size"]) * int(data["scan_mag"])/int(data["tile_mag"])

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

    annotation_name = data["column"] + " tile-based heatmap"

    save_push_results(base_dsa_annotation, elements, annotation_name, data["image_name"],
                      ctx.obj['uri'], ctx.obj['token'])


if __name__ == '__main__':
    cli()
