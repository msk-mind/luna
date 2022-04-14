import click
from decimal import Decimal
import pandas as pd
import json
import geojson
import ijson
import copy
import os
import logging
from PIL import Image
import re
import numpy as np

from luna.pathology.dsa.utils import (
    get_continuous_color,
    vectorize_np_array_bitmask_by_pixel_value,
)
from luna.common.utils import cli_runner
from luna.common.custom_logger import init_logger

init_logger()
logger = logging.getLogger("dsa_viz")

# Base DSA jsons
base_dsa_polygon_element = {
    "fillColor": "rgba(0, 0, 0, 0)",
    "lineColor": "rgb(0, 0, 0)",
    "lineWidth": 2,
    "type": "polyline",
    "closed": True,
    "points": [],
    "label": {"value": ""},
}
base_dsa_point_element = {
    "fillColor": "rgba(0, 0, 0, 0)",
    "lineColor": "rgb(0, 0, 0)",
    "lineWidth": 2,
    "type": "point",
    "center": [],
    "label": {"value": ""},
}
base_dsa_annotation = {"description": "", "elements": [], "name": ""}

# Qupath 20x mag factor
QUPATH_MAG_FACTOR = 0.5011
image_id_regex = "(.*).svs"


def check_filepaths_valid(filepaths):
    """Checks if all paths exist.

    Args:
        filepaths (list): file paths

    Returns:
        bool: True if all file paths exist, False otherwise
    """

    all_files_found = True
    for filepath in filepaths:
        if not os.path.exists(filepath):
            logger.warning(f"Filepath in config: {filepath} does not exist")
            all_files_found = False
    return all_files_found


def save_dsa_annotation(
    base_annotation, elements, annotation_name, output_dir, image_filename
):
    """Helper function to save annotation elements to a json file.

    Args:
        base_annotation (dict): base annotation structure for DSA
        elements (list): list of annotation elements
        annotation_name (string): annotation name for HistomicsUI
        output_dir (string): path to a directory to save the annotation file
        image_filename (string): name of the image in DSA e.g. 123.svs

    Returns:
        string: annotation file path. None if error in writing the file.
    """
    dsa_annotation = copy.deepcopy(base_annotation)

    dsa_annotation["elements"] = elements
    dsa_annotation["name"] = annotation_name

    image_id = re.search(image_id_regex, image_filename).group(1)
    annotation_name_replaced = annotation_name.replace(" ", "_")

    os.makedirs(output_dir, exist_ok=True)
    outfile_name = os.path.join(
        output_dir, f"{annotation_name_replaced}_{image_id}.json"
    )

    try:
        with open(outfile_name, "w") as outfile:
            json.dump(dsa_annotation, outfile)
        return outfile_name
    except Exception as exc:
        logger.error(exc)
        return None


@click.group()
def cli():
    """Convert segmentations, bitmasks, heatmaps to DSA annotation Json
    format."""
    pass


@click.argument("input", nargs=1)
@click.option(
    "-o",
    "--output_dir",
    help="directory to save the DSA compatible annotation json",
    required=False,
)
@click.option(
    "-f",
    "--image_filename",
    help="name of the image file in DSA e.g. 123.svs",
    required=False,
)
@click.option(
    "-a",
    "--annotation_name",
    help="name of the annotation to be displayed in DSA",
    required=False,
)
@click.option(
    "-lc",
    "--line_colors",
    help="user-provided line color map with {feature name:rgb values}",
    required=False,
)
@click.option(
    "-fc",
    "--fill_colors",
    help="user-provided line color map with {feature name:rgba values}",
    required=False,
)
@click.option(
    "-m",
    "--method_param_path",
    required=False,
    help="path to a metadata json/yaml file with method parameters to "
    "reproduce results",
)
@cli.command()
def stardist_polygon(**cli_kwargs):
    """
    Example:

        \b
        dsa stardist-polygon
            ../dsa_input/test_object_classification.geojson
            --output_dir ../dsa_annotations/stardist_polygon
            --annotation_name stardist_polygon_segmentations
            --image_filename 123.svs
            --line_colors '{"Other": "rgb(0,255,0)", "Lymphocyte": "rgb(255,0,0)"}'
            --fill_colors '{"Other": "rgba(0,255,0,100)", "Lymphocyte": "rgba(255,0,0,100)"}'
    """
    params = [
        ("fill_colors", dict),
        ("line_colors", dict),
        ("annotation_name", str),
        ("image_filename", str),
        ("output_dir", str),
        ("input", str),
    ]
    cli_runner(cli_kwargs, params, stardist_polygon_main)


def stardist_polygon_main(
    input, output_dir, image_filename, annotation_name, line_colors, fill_colors
):
    """Build DSA annotation json from stardist geojson classification results

    Args:
        input (string): path to stardist geojson classification results
        output_dir (string): directory to save the DSA compatible annotation
        json
        image_filename (string): name of the image file in DSA e.g. 123.svs
        annotation_name (string): name of the annotation to be displayed in DSA
        line_colors (map, optional): user-provided line color map with {
        feature name:rgb values}
        fill_colors (map, optional): user-provided fill color map with {
        feature name:rgba values}

    Returns:
        dict: annotation file path
    """
    # TODO: find better fix
    # can't handle NaNs for vectors, do this to replace all NaNs
    # for now: https://stackoverflow.com/questions/17140886/how-to-search
    # -and-replace-text-in-a-file
    with open(input, "r") as input_file:
        filedata = input_file.read()
    newdata = filedata.replace("NaN", "-1")

    elements = []
    for cell in ijson.items(newdata, "item"):
        label_name = cell["properties"]["classification"]["name"]
        coord_list = list(cell["geometry"]["coordinates"][0])

        # uneven nested list when iterative parsing of json --> make sure
        # to get the list of coords
        # this can come as mixed types as well, so type checking needed
        while (
            isinstance(coord_list, list)
            and isinstance(coord_list[0], list)
            and not isinstance(coord_list[0][0], (int, float, Decimal))
        ):
            coord_list = coord_list[0]

        coords = [[float(coord[0]), float(coord[1]), 0] for coord in coord_list]
        element = copy.deepcopy(base_dsa_polygon_element)

        element["label"]["value"] = label_name
        element["fillColor"] = fill_colors[label_name]
        element["lineColor"] = line_colors[label_name]
        element["points"] = coords

        elements.append(element)

    annotatation_filepath = save_dsa_annotation(
        base_dsa_annotation, elements, annotation_name, output_dir, image_filename
    )
    return {"dsa_annotaiton": annotatation_filepath}


@click.argument("input", nargs=1)
@click.option(
    "-o",
    "--output_dir",
    help="directory to save the DSA compatible annotation json",
    required=False,
)
@click.option(
    "-f",
    "--image_filename",
    help="name of the image file in DSA e.g. 123.svs",
    required=False,
)
@click.option(
    "-a",
    "--annotation_name",
    help="name of the annotation to be displayed in DSA",
    required=False,
)
@click.option(
    "-lc",
    "--line_colors",
    help="user-provided line color map with {feature name:rgb values}",
    required=False,
)
@click.option(
    "-fc",
    "--fill_colors",
    help="user-provided line color map with {feature name:rgba values}",
    required=False,
)
@click.option(
    "-m",
    "--method_param_path",
    required=False,
    help="path to a metadata json/yaml file with method parameters to "
    "reproduce results",
)
@cli.command()
def stardist_cell(**cli_kwargs):
    """
    Example:

        \b
        dsa stardist-cell
            ../dsa_input/test_object_detection.tsv
            --output_dir ../dsa_annotations/stardist_cell
            --annotation_name stardist_cell_segmentations
            --image_filename 123.svs
            --line_colors '{"Other": "rgb(0,255,0)", "Lymphocyte": "rgb(255,0,0)"}'
            --fill_colors '{"Other": "rgba(0,255,0,100)", "Lymphocyte": "rgba(255,0,0,100)"}'
    """
    params = [
        ("fill_colors", dict),
        ("line_colors", dict),
        ("annotation_name", str),
        ("image_filename", str),
        ("output_dir", str),
        ("input", str),
    ]
    cli_runner(cli_kwargs, params, stardist_cell_main)


def stardist_cell_main(
    input, output_dir, image_filename, annotation_name, line_colors, fill_colors
):
    """Build DSA annotation json from TSV classification data generated by
    stardist

    Processes a cell classification data generated by Qupath/stardist and
    adds the center coordinates of the cells
    as annotation elements.

    Args:
        input (string): path to TSV classification data generated by stardist
        output_dir (string): directory to save the DSA compatible annotation
        json
        image_filename (string): name of the image file in DSA e.g. 123.svs
        annotation_name (string): name of the annotation to be displayed in DSA
        line_colors (map, optional): line color map with {feature name:rgb
        values}
        fill_colors (map, optional): fill color map with {feature name:rgba
        values}

    Returns:
        dict: annotation file path
    """
    # qupath_stardist_cell_tsv can be quite large to load all columns
    # into memory (contains many feature columns),
    # so only load baisc columns that are needed for now
    cols_to_load = [
        "Image",
        "Name",
        "Class",
        "ROI",
        "Centroid X µm",
        "Centroid Y µm",
        "Parent",
    ]
    df = pd.read_csv(input, sep="\t", usecols=cols_to_load, index_col=False)

    # do some preprocessing on the tsv -- e.g. stardist sometimes finds
    # cells in glass
    df = df[df["Parent"] != "Glass"]
    # populate json elements
    elements = []
    for idx, row in df.iterrows():
        elements_entry = copy.deepcopy(base_dsa_point_element)

        # x,y coordinates from stardist are in microns so divide by
        # QUPATH_MAG_FACTOR = 0.5011 (exact 20x mag factor used by qupath
        # specifically)
        x = row["Centroid X µm"] / QUPATH_MAG_FACTOR
        y = row["Centroid Y µm"] / QUPATH_MAG_FACTOR

        # Get cell label and add to element
        label_name = row["Class"]
        elements_entry["label"]["value"] = label_name
        elements_entry["fillColor"] = fill_colors[label_name]
        elements_entry["lineColor"] = line_colors[label_name]

        # add centroid coordinate of cell to element
        center = [x, y, 0]
        elements_entry["center"] = center

        elements.append(elements_entry)

    annotatation_filepath = save_dsa_annotation(
        base_dsa_annotation, elements, annotation_name, output_dir, image_filename
    )
    return {"dsa_annotation": annotatation_filepath}


@click.argument("input", nargs=1)
@click.option(
    "-o",
    "--output_dir",
    help="directory to save the DSA compatible annotation json",
    required=False,
)
@click.option(
    "-f",
    "--image_filename",
    help="name of the image file in DSA e.g. 123.svs",
    required=False,
)
@click.option(
    "-a",
    "--annotation_name",
    help="name of the annotation to be displayed in DSA",
    required=False,
)
@click.option(
    "-lc",
    "--line_colors",
    help="user-provided line color map with {feature name:rgb values}",
    required=False,
)
@click.option(
    "-fc",
    "--fill_colors",
    help="user-provided line color map with {feature name:rgba values}",
    required=False,
)
@click.option(
    "-m",
    "--method_param_path",
    required=False,
    help="path to a metadata json/yaml file with method parameters to "
    "reproduce results",
)
@cli.command()
def regional_polygon(**cli_kwargs):
    """
    Example:

        \b
        dsa regional-polygon
            ../dsa_input/regional_annotation.json
            --output_dir ../dsa_annotations/regional_polygon
            --annotation_name regional_polygon
            --image_filename 123.svs
            --line_colors '{"Other": "rgb(0,255,0)", "Lymphocyte": "rgb(255,0,0)"}'
            --fill_colors '{"Other": "rgba(0,255,0,100)", "Lymphocyte": "rgba(255,0,0,100)"}'
    """
    params = [
        ("fill_colors", dict),
        ("line_colors", dict),
        ("annotation_name", str),
        ("image_filename", str),
        ("output_dir", str),
        ("input", str),
    ]
    cli_runner(cli_kwargs, params, regional_polygon_main)


def regional_polygon_main(
    input, output_dir, image_filename, annotation_name, line_colors, fill_colors
):
    """Build DSA annotation json from regional annotation geojson

    Args:
        input (string): path to regional annotation geojson
        output_dir (string): directory to save the DSA compatible annotation
        json
        image_filename (string): name of the image file in DSA e.g. 123.svs
        annotation_name (string): name of the annotation to be displayed in DSA
        line_colors (map, optional): line color map with {feature name:rgb
        values}
        fill_colors (map, optional): fill color map with {feature name:rgba
        values}

    Returns:
        dict: annotation file path
    """
    with open(input) as regional_file:
        regional_annotation = geojson.loads(geojson.load(regional_file))

    elements = []
    for annot in regional_annotation["features"]:

        # get label name and add to element
        element = copy.deepcopy(base_dsa_polygon_element)
        label_name = annot.properties["label_name"]
        element["label"]["value"] = label_name
        element["fillColor"] = fill_colors[label_name]
        element["lineColor"] = line_colors[label_name]

        # add coordinates
        coords = annot["geometry"]["coordinates"]
        # if coordinates have extra nesting, set coordinates to 2d array.
        coords_arr = np.array(coords)
        if coords_arr.ndim == 3 and coords_arr.shape[0] == 1:
            coords = np.squeeze(coords_arr).tolist()

        for c in coords:
            c.append(0)
        element["points"] = coords
        elements.append(element)

    annotatation_filepath = save_dsa_annotation(
        base_dsa_annotation, elements, annotation_name, output_dir, image_filename
    )
    return {"dsa_annotation": annotatation_filepath}


@click.argument("input", nargs=1)
@click.option(
    "-o",
    "--output_dir",
    help="directory to save the DSA compatible annotation json",
    required=False,
)
@click.option(
    "-f",
    "--image_filename",
    help="name of the image file in DSA e.g. 123.svs",
    required=False,
)
@click.option(
    "-a",
    "--annotation_name",
    help="name of the annotation to be displayed in DSA",
    required=False,
)
@click.option(
    "-cl",
    "--classes_to_include",
    help="list of classification labels to visualize",
    required=False,
)
@click.option(
    "-lc",
    "--line_colors",
    help="user-provided line color map with {feature name:rgb values}",
    required=False,
)
@click.option(
    "-fc",
    "--fill_colors",
    help="user-provided line color map with {feature name:rgba values}",
    required=False,
)
@click.option(
    "-m",
    "--method_param_path",
    required=False,
    help="path to a metadata json/yaml file with method parameters to "
    "reproduce results",
)
@cli.command()
def qupath_polygon(**cli_kwargs):
    """
    Example:

        \b
        dsa qupath-polygon
            ../dsa_input/quppath.geojson
            --output_dir ../dsa_annotations/quppath
            --annotation_name quppath
            --image_filename 123.svs
            --classes_to_include Tumor,Other
            --line_colors '{"Other": "rgb(0,255,0)", "Lymphocyte": "rgb(255,0,0)"}'
            --fill_colors '{"Other": "rgba(0,255,0,100)", "Lymphocyte": "rgba(255,0,0,100)"}'
    """
    params = [
        ("fill_colors", dict),
        ("line_colors", dict),
        ("annotation_name", str),
        ("image_filename", str),
        ("output_dir", str),
        ("input", str),
        ("classes_to_include", list),
    ]
    cli_runner(cli_kwargs, params, qupath_polygon_main)


def qupath_polygon_main(
    input,
    output_dir,
    image_filename,
    annotation_name,
    classes_to_include,
    line_colors,
    fill_colors,
):
    """Build DSA annotation json from Qupath polygon geojson

    Args:
        input (string): path to Qupath polygon geojson
        output_dir (string): directory to save the DSA compatible annotation
        json
        image_filename (string): name of the image file in DSA e.g. 123.svs
        annotation_name (string): name of the annotation to be displayed in DSA
        classes_to_include (list): list of classification labels to visualize
        e.g. ["Tumor", "Stroma", ...]
        line_colors (map, optional): line color map with {feature name:rgb
        values}
        fill_colors (map, optional): fill color map with {feature name:rgba
        values}

    Returns:
        dict: annotation file path
    """
    with open(input) as regional_file:
        pixel_clf_polygons = geojson.load(regional_file)

    elements = []
    for polygon in pixel_clf_polygons:

        props = polygon.properties
        if "classification" not in props:
            continue

        label_name = polygon.properties["classification"]["name"]
        if label_name in classes_to_include:

            element = copy.deepcopy(base_dsa_polygon_element)
            element["label"]["value"] = label_name
            element["fillColor"] = fill_colors[label_name]
            element["lineColor"] = line_colors[label_name]

            coords = polygon["geometry"]["coordinates"]

            # uneven nesting of connected components
            for coord in coords:
                if isinstance(coord[0], list) and isinstance(coord[0][0], (int, float)):
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

                        connected_component_element[
                            "points"
                        ] = connected_component_coords
                        elements.append(connected_component_element)

    annotatation_filepath = save_dsa_annotation(
        base_dsa_annotation, elements, annotation_name, output_dir, image_filename
    )
    return {"dsa_annotation": annotatation_filepath}


@click.argument("input", nargs=1)
@click.option(
    "-o",
    "--output_dir",
    help="directory to save the DSA compatible annotation json",
    required=False,
)
@click.option(
    "-f",
    "--image_filename",
    help="name of the image file in DSA e.g. 123.svs",
    required=False,
)
@click.option(
    "-a",
    "--annotation_name",
    help="name of the annotation to be displayed in DSA",
    required=False,
)
@click.option(
    "-lc",
    "--line_colors",
    help="user-provided line color map with {feature name:rgb values}",
    required=False,
)
@click.option(
    "-fc",
    "--fill_colors",
    help="user-provided line color map with {feature name:rgba values}",
    required=False,
)
@click.option(
    "-m",
    "--method_param_path",
    required=False,
    help="path to a metadata json/yaml file with method parameters to "
    "reproduce results",
)
@cli.command()
def bitmask_polygon(**cli_kwargs):
    """
    Example:

        \b
        dsa bitmask-polygon
            '{"Tumor": "non/existing/path/to/png.png"}'
            --output_dir ../dsa_annotations/bitmask
            --annotation_name bitmask
            --image_filename 123.svs
            --line_colors '{"Other": "rgb(0,255,0)", "Lymphocyte": "rgb(255,0,0)"}'
            --fill_colors '{"Other": "rgba(0,255,0,100)", "Lymphocyte": "rgba(255,0,0,100)"}'
    """
    params = [
        ("fill_colors", dict),
        ("line_colors", dict),
        ("annotation_name", str),
        ("image_filename", str),
        ("output_dir", str),
        ("input", dict),
    ]
    cli_runner(cli_kwargs, params, bitmask_polygon_main)


def bitmask_polygon_main(
    input, output_dir, image_filename, annotation_name, line_colors, fill_colors
):
    """Build DSA annotation json from bitmask PNGs

    Vectorizes and simplifies contours from the bitmask.

    Args:
        input (map): map of {label:path_to_bitmask_png}
        output_dir (string): directory to save the DSA compatible annotation
        json
        image_filename (string): name of the image file in DSA e.g. 123.svs
        annotation_name (string): name of the annotation to be displayed in DSA
        line_colors (map, optional): line color map with {feature name:rgb
        values}
        fill_colors (map, optional): fill color map with {feature name:rgba
        values}

    Returns:
        dict: annotation file path
    """
    if not check_filepaths_valid(input.values()):
        raise ValueError("No valid PNG masks found. Exiting..")

    elements = []
    for bitmask_label, bitmask_filepath in input.items():
        Image.MAX_IMAGE_PIXELS = 5000000000
        annotation = Image.open(bitmask_filepath)
        bitmask_np = np.array(annotation)
        simplified_contours = vectorize_np_array_bitmask_by_pixel_value(bitmask_np)

        for n, contour in enumerate(simplified_contours):
            element = copy.deepcopy(base_dsa_polygon_element)
            label_name = bitmask_label
            element["label"]["value"] = label_name
            element["fillColor"] = fill_colors[label_name]
            element["lineColor"] = line_colors[label_name]

            coords = contour.tolist()
            for c in coords:
                c.append(0)
            element["points"] = coords
            elements.append(element)

    annotatation_filepath = save_dsa_annotation(
        base_dsa_annotation, elements, annotation_name, output_dir, image_filename
    )
    return {"dsa_annotation": annotatation_filepath}


@click.argument("input", nargs=1)
@click.option(
    "-o",
    "--output_dir",
    help="directory to save the DSA compatible annotation json",
    required=False,
)
@click.option(
    "-f",
    "--image_filename",
    help="name of the image file in DSA e.g. 123.svs",
    required=False,
)
@click.option(
    "-a",
    "--annotation_name",
    help="name of the annotation to be displayed in DSA",
    required=False,
)
@click.option(
    "-c",
    "--column",
    help="column to visualize e.g. tile_score",
    required=False,
)
@click.option(
    "-ts",
    "--tile_size",
    help="tile size",
    required=False,
)
@click.option(
    "-sc",
    "--scale_factor",
    help="scale to match image DSA. (default 1)",
    default=1,
    required=False,
)
@click.option(
    "-m",
    "--method_param_path",
    required=False,
    help="path to a metadata json/yaml file with method parameters to "
    "reproduce results",
)
@cli.command()
def heatmap(**cli_kwargs):
    """
    Example:

        \b
        dsa heatmap
            score.csv
            --output_dir ../dsa_annotations/heatmap
            --annotation_name heatmap
            --image_filename 123.svs
            --tile_size 256
            --column tumor
            --scale_factor 1
    """
    params = [
        ("column", str),
        ("tile_size", int),
        ("annotation_name", str),
        ("image_filename", str),
        ("output_dir", str),
        ("input", str),
        ("scale_factor", int),
    ]
    cli_runner(cli_kwargs, params, heatmap_main)


def heatmap_main(
    input, output_dir, image_filename, annotation_name, column, tile_size, scale_factor
):
    """Generate heatmap based on the tile scores

    Creates a heatmap for the given column, using the color palette `viridis`
    to set a fill value
    - the color ranges from purple to yellow, for scores from 0 to 1.

    Args:
        input (string): path to CSV with tile scores
        output_dir (string): directory to save the DSA compatible annotation
        json
        image_filename (string): name of the image file in DSA e.g. 123.svs
        annotation_name (string): name of the annotation to be displayed in DSA
        column (string): column to visualize e.g. tile_score
        tile_size (int): size of tiles
        scale_factor (int, optional): scale to match the image on DSA. By
        default, 1.

    Returns:
        string: annotation file path. None if error in writing the file.
    """
    df = pd.read_csv(input)
    scaled_tile_size = int(tile_size * int(scale_factor if scale_factor else 1))

    elements = []
    for _, row in df.iterrows():
        element = copy.deepcopy(base_dsa_polygon_element)
        label_value = row[column]
        element["label"]["value"] = str(label_value)

        # get label specific color and add to elements
        line_colors, fill_colors = get_continuous_color(label_value)
        element["fillColor"] = fill_colors
        element["lineColor"] = line_colors

        # convert coordinate string to tuple using eval
        x, y = eval(row["coordinates"])

        pixel_x = x * scaled_tile_size
        pixel_y = y * scaled_tile_size

        coords = [
            [pixel_x, pixel_y],
            [pixel_x + scaled_tile_size, pixel_y],
            [pixel_x + scaled_tile_size, pixel_y + scaled_tile_size],
            [pixel_x, pixel_y + scaled_tile_size],
            [pixel_x, pixel_y],
        ]
        for c in coords:
            c.append(0)
        element["points"] = coords
        elements.append(element)

    annotation_name = column + "_" + annotation_name

    annotatation_filepath = save_dsa_annotation(
        base_dsa_annotation, elements, annotation_name, output_dir, image_filename
    )
    return {"dsa_annotation": annotatation_filepath}


@click.argument("input", nargs=1)
@click.option("-l", "--label", help="map of {label_num:label_name}", required=False)
@click.option(
    "-o",
    "--output_dir",
    help="directory to save the DSA compatible annotation json",
    required=False,
)
@click.option(
    "-f",
    "--image_filename",
    help="name of the image file in DSA e.g. 123.svs",
    required=False,
)
@click.option(
    "-a",
    "--annotation_name",
    help="name of the annotation to be displayed in DSA",
    required=False,
)
@click.option(
    "-lc",
    "--line_colors",
    help="user-provided line color map with {feature name:rgb values}",
    required=False,
)
@click.option(
    "-fc",
    "--fill_colors",
    help="user-provided line color map with {feature name:rgba values}",
    required=False,
)
@click.option(
    "-sc",
    "--scale_factor",
    help="scale to match image DSA. (default 1)",
    required=False,
)
@click.option(
    "-m",
    "--method_param_path",
    required=False,
    help="path to a metadata json/yaml file with method parameters to "
    "reproduce results",
)
@cli.command()
def bmp_polygon(**cli_kwargs):
    """
    Example:

        \b
        dsa bmp-polygon
            results.bmp
            --output_dir ../dsa_annotations/bmp
            --annotation_name bmp
            --image_filename 123.svs
            --label '{0: "Tumor", 1: "Other"}'
            --scale_factor 1
            --line_colors '{"Other": "rgb(0,255,0)", "Tumor": "rgb(255,0,0)"}'
            --fill_colors '{"Other": "rgba(0,255,0,100)", "Tumor": "rgba(255,0,0,100)"}'
    """
    params = [
        ("fill_colors", dict),
        ("line_colors", dict),
        ("annotation_name", str),
        ("image_filename", str),
        ("output_dir", str),
        ("input", str),
        ("label", dict),
        ("scale_factor", int),
    ]
    cli_runner(cli_kwargs, params, bmp_polygon_main)


def bmp_polygon_main(
    label,
    input,
    output_dir,
    image_filename,
    annotation_name,
    line_colors,
    fill_colors,
    scale_factor,
):
    """Build DSA annotation json from a BMP with multiple labels.

    Vectorizes and simplifies contours per label.

    Args:
        input (string): path to bmp file
        output_dir (string): directory to save the DSA compatible annotation
        json
        image_filename (string): name of the image file in DSA e.g. 123.svs
        annotation_name (string): name of the annotation to be displayed in DSA
        line_colors (map): line color map with {feature name:rgb
        values}
        fill_colors (map): fill color map with {feature name:rgba
        values}
        scale_factor (int, optional): scale to match image DSA. (default 1)

    Returns:
        dict: annotation file path
    """
    elements = []
    Image.MAX_IMAGE_PIXELS = 5000000000
    annotation = Image.open(input)
    arr = np.array(annotation)

    for label_num, label_name in label.items():
        simplified_contours = vectorize_np_array_bitmask_by_pixel_value(
            arr, label_num, scale_factor=scale_factor
        )

        for n, contour in enumerate(simplified_contours):
            element = copy.deepcopy(base_dsa_polygon_element)
            element["label"]["value"] = label_name
            element["fillColor"] = fill_colors[label_name]
            element["lineColor"] = line_colors[label_name]

            coords = contour.tolist()
            for c in coords:
                c.append(0)
            element["points"] = coords
            elements.append(element)

    annotatation_filepath = save_dsa_annotation(
        base_dsa_annotation, elements, annotation_name, output_dir, image_filename
    )
    return {"dsa_annotation": annotatation_filepath}


if __name__ == "__main__":
    cli()
