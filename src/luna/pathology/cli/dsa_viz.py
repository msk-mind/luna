import copy
import json
import logging
import re
from decimal import Decimal
from pathlib import Path
from typing import Optional

import fire  # type: ignore
import fsspec  # type: ignore
import geojson  # type: ignore
import ijson  # type: ignore
import numpy as np
import pandas as pd
from fsspec import open  # type: ignore
from PIL import Image
from typing_extensions import TypedDict

from luna.common.custom_logger import init_logger
from luna.common.utils import get_config, save_metadata, timed
from luna.pathology.dsa.utils import (
    get_continuous_color,
    vectorize_np_array_bitmask_by_pixel_value,
)

init_logger()
logger = logging.getLogger("dsa_viz")

# Base DSA jsons
PolygonElement = TypedDict(
    "PolygonElement",
    {
        "fillColor": str,
        "lineColor": str,
        "lineWidth": int,
        "type": str,
        "closed": bool,
        "points": list,
        "label": dict[str, str],
    },
)
base_dsa_polygon_element: PolygonElement = {
    "fillColor": "rgba(0, 0, 0, 0)",
    "lineColor": "rgb(0, 0, 0)",
    "lineWidth": 2,
    "type": "polyline",
    "closed": True,
    "points": [],
    "label": {"value": ""},
}
PointElement = TypedDict(
    "PointElement",
    {
        "fillColor": str,
        "lineColor": str,
        "lineWidth": int,
        "type": str,
        "center": list,
        "label": dict[str, str],
    },
)
base_dsa_point_element: PointElement = {
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


class InvalidImageIdException(BaseException):
    pass


def check_filepaths_valid(urls, storage_options):
    """Checks if all paths exist.

    Args:
        filepaths (list): file paths

    Returns:
        bool: True if all file paths exist, False otherwise
    """

    all_files_found = True
    for url in urls:
        fs, urlpath = fsspec.core.url_to_fs(url, **storage_options)
        if not fs.exists(urlpath):
            logger.warning(f"url in config: {url} does not exist")
            all_files_found = False
    return all_files_found


def get_dsa_annotation(elements: list, annotation_name: str, description: str = ""):
    """Helper function to get dsa annotation

    Args:
        elements (list): list of annotation elements
        annotation_name (string): annotation name for HistomicsUI
        image_filename (string): name of the image in DSA e.g. 123.svs

    Returns:
        string: annotation file path. None if error in writing the file.
    """
    dsa_annotation = {
        "description": description,
        "elements": elements,
        "name": annotation_name,
    }

    dsa_annotation["elements"] = elements
    dsa_annotation["name"] = annotation_name

    return dsa_annotation


def save_dsa_annotation(
    dsa_annotation: dict,
    output_url_prefix: str,
    image_filename: str,
    storage_options: dict,
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

    result = re.search(image_id_regex, image_filename)
    if result:
        image_id = result.group(1)
    else:
        raise InvalidImageIdException(f"Invalid image filename: {image_filename}")

    annotation_name_replaced = dsa_annotation["name"].replace(" ", "_")

    fs, output_urlpath_prefix = fsspec.core.url_to_fs(
        output_url_prefix, **storage_options
    )
    output_url = (
        Path(output_urlpath_prefix) / f"{annotation_name_replaced}_{image_id}.json"
    )

    try:
        with open(output_url, "w", **storage_options).open() as outfile:
            json.dump(dsa_annotation, outfile)
        return output_url
    except Exception as exc:
        logger.error(exc)
        return None


@timed
@save_metadata
def stardist_polygon(
    input_url: str = "???",
    image_filename: str = "???",
    annotation_name: str = "???",
    output_url: str = "???",
    line_colors: dict[str, str] = {},
    fill_colors: dict[str, str] = {},
    storage_options: dict = {},
    local_config: str = "",
):
    """Build DSA annotation json from stardist geojson classification results

    Args:
        input_url (string): url to stardist geojson classification results json
        image_filename (string): name of the image file in DSA e.g. 123.svs
        annotation_name (string): name of the annotation to be displayed in DSA
        output_url (string): URL prefix to save annotations
        line_colors (dict): user-provided line color map with {
        feature name:rgb values}
        fill_colors (dict): user-provided fill color map with {
        feature name:rgba values}
        storage_options (dict): storage options to pass to read/write functions
        local_config (string): local config YAML file

    Returns:
        dict: annotation file path
    """
    """
    Example:

        \b
        dsa stardist-polygon
            --input_url ../dsa_input/test_object_classification.geojson
            --output_url ../dsa_annotations/stardist_polygon
            --annotation_name stardist_polygon_segmentations
            --image_filename 123.svs
            --line_colors '{"Other": "rgb(0,255,0)", "Lymphocyte": "rgb(255,0,0)"}'
            --fill_colors '{"Other": "rgba(0,255,0,100)", "Lymphocyte": "rgba(255,0,0,100)"}'
    """
    config = get_config(vars())
    dsa_annotation = stardist_polygon_main(
        config["input_url"],
        config["annotation_name"],
        config["line_colors"],
        config["fill_colors"],
        config["storage_options"],
    )
    annotatation_filepath = save_dsa_annotation(
        dsa_annotation,
        config["output_url"],
        config["image_filename"],
        config["storage_options"],
    )
    return {"dsa_annotation": annotatation_filepath}


def stardist_polygon_main(
    input_url: str,
    annotation_name: str,
    line_colors: dict[str, str],
    fill_colors: dict[str, str],
    storage_options: dict,
):
    """Build DSA annotation from stardist geojson classification results

    Args:
        input_url (string): URL to stardist geojson classification results
        json
        annotation_name (string): name of the annotation to be displayed in DSA
        line_colors (dict): user-provided line color map with {
        feature name:rgb values}
        fill_colors (dict): user-provided fill color map with {
        feature name:rgba values}

    Returns:
        dict: annotation file path
    """
    # TODO: find better fix
    # can't handle NaNs for vectors, do this to replace all NaNs
    # for now: https://stackoverflow.com/questions/17140886/how-to-search
    # -and-replace-text-in-a-file
    with open(input_url, "r", **storage_options).open() as input_file:
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

        element["label"]["value"] = str(label_name)
        element["fillColor"] = fill_colors[label_name]
        element["lineColor"] = line_colors[label_name]
        element["points"] = coords

        elements.append(element)

    return get_dsa_annotation(elements, annotation_name)


@timed
@save_metadata
def stardist_cell(
    input_url: str = "???",
    output_url: str = "???",
    image_filename: str = "???",
    annotation_name: str = "???",
    line_colors: Optional[dict[str, str]] = None,
    fill_colors: Optional[dict[str, str]] = None,
    storage_options: dict = {},
    local_config: str = "",
):
    """Build DSA annotation json from TSV classification data generated by
    stardist

    Processes a cell classification data generated by Qupath/stardist and
    adds the center coordinates of the cells
    as annotation elements.

    Args:
        input_url (string): path to TSV classification data generated by stardist
        output_url (string): URL prefix for saving dsa annotation json
        image_filename (string): name of the image file in DSA e.g. 123.svs
        annotation_name (string): name of the annotation to be displayed in DSA
        line_colors (dict, optional): line color map with {feature name:rgb
        values}
        fill_colors (dict, optional): fill color map with {feature name:rgba
        values}
        storage_options (dict): storage options to pass to read/write functions
        local_config (string): local config YAML file

    Returns:
        dict: annotation file path
    """
    config = get_config(vars())
    dsa_annotation = stardist_cell_main(
        config["input_url"],
        config["image_filename"],
        config["annotation_name"],
        config["line_colors"],
        config["fill_colors"],
        config["storage_options"],
    )
    annotatation_filepath = save_dsa_annotation(
        dsa_annotation,
        config["output_url"],
        config["image_filename"],
        config["storage_options"],
    )
    return {"dsa_annotation": annotatation_filepath}


def stardist_cell_main(
    input_url: str,
    image_filename: str,
    annotation_name: str,
    line_colors: Optional[dict[str, str]],
    fill_colors: Optional[dict[str, str]],
    storage_options: dict,
):
    """Build DSA annotation json from TSV classification data generated by
    stardist

    Processes a cell classification data generated by Qupath/stardist and
    adds the center coordinates of the cells
    as annotation elements.

    Args:
        input_url (string): url to TSV classification data generated by stardist
        image_filename (string): name of the image file in DSA e.g. 123.svs
        annotation_name (string): name of the annotation to be displayed in DSA
        line_colors (dict, optional): line color map with {feature name:rgb
        values}
        fill_colors (dict, optional): fill color map with {feature name:rgba
        values}
        storage_options (dict): storage options to pass to read/write functions

    Returns:
        dict: dsa annotation
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
    df = pd.read_csv(
        input_url,
        sep="\t",
        usecols=cols_to_load,
        index_col=False,
        storage_options=storage_options,
    )

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
        if fill_colors:
            elements_entry["fillColor"] = fill_colors[label_name]
        if line_colors:
            elements_entry["lineColor"] = line_colors[label_name]

        # add centroid coordinate of cell to element
        center = [x, y, 0]
        elements_entry["center"] = center

        elements.append(elements_entry)

    return get_dsa_annotation(elements, annotation_name)


@timed
@save_metadata
def regional_polygon(
    input_url: str = "???",
    output_url: str = "???",
    image_filename: str = "???",
    annotation_name: str = "???",
    line_colors: Optional[dict[str, str]] = None,
    fill_colors: Optional[dict[str, str]] = None,
    storage_options: dict = {},
    local_config: str = "",
):
    """Build DSA annotation json from regional annotation geojson

    Args:
        input_url (string): url of to regional annotation geojson
        output_url (string): URL prefix for saving dsa annotation json
        annotation_name (string): name of the annotation to be displayed in DSA
        line_colors (dict, optional): line color map with {feature name:rgb
        values}
        fill_colors (dict, optional): fill color map with {feature name:rgba
        values}
        storage_options (dict): storage options to pass to read/write functions
        local_config (string): local config yaml file

    Returns:
        dict: annotation file path
    """

    config = get_config(vars())

    dsa_annotation = regional_polygon_main(
        config["input_url"],
        config["annotation_name"],
        config["line_colors"],
        config["fill_colors"],
        config["storage_options"],
    )

    annotatation_filepath = save_dsa_annotation(
        dsa_annotation,
        config["output_url"],
        config["image_filename"],
        config["storage_options"],
    )
    return {"dsa_annotation": annotatation_filepath}


def regional_polygon_main(
    input_url: str,
    annotation_name: str,
    line_colors: Optional[dict[str, str]],
    fill_colors: Optional[dict[str, str]],
    storage_options: dict,
):
    """Build DSA annotation json from regional annotation geojson

    Args:
        input (string): path to regional annotation geojson
        annotation_name (string): name of the annotation to be displayed in DSA
        line_colors (dict, optional): line color map with {feature name:rgb
        values}
        fill_colors (dict, optional): fill color map with {feature name:rgba
        values}
        storage_options (dict): storage options to pass to read/write functions

    Returns:
        dict: DSA annotation
    """
    with open(input_url, **storage_options).open() as regional_file:
        regional_annotation = geojson.loads(geojson.load(regional_file))

    elements = []
    for annot in regional_annotation["features"]:
        # get label name and add to element
        element = copy.deepcopy(base_dsa_polygon_element)
        label_name = annot.properties["label_name"]
        element["label"]["value"] = label_name
        if fill_colors:
            element["fillColor"] = fill_colors[label_name]
        if line_colors:
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

    return get_dsa_annotation(elements, annotation_name)


@timed
@save_metadata
def qupath_polygon(
    input_url: str = "???",
    output_url: str = "???",
    image_filename: str = "???",
    annotation_name: str = "???",
    classes_to_include: list = "???",  # type: ignore
    line_colors: Optional[dict[str, str]] = None,
    fill_colors: Optional[dict[str, str]] = None,
    storage_options: dict = {},
    local_config: str = "",
):
    """Build DSA annotation json from Qupath polygon geojson

    Args:
        input_url (string): URL of Qupath polygon geojson
        output_url (string): URL prefix for saving the DSA compatible annotation
        json
        image_filename (string): name of the image file in DSA e.g. 123.svs
        annotation_name (string): name of the annotation to be displayed in DSA
        classes_to_include (list): list of classification labels to visualize
        e.g. ["Tumor", "Stroma", ...]
        line_colors (dict, optional): line color map with {feature name:rgb
        values}
        fill_colors (dict, optional): fill color map with {feature name:rgba
        values}
        storage_options (dict): storage options to pass to read/write functions
        local_config (string): local config yaml file

    Returns:
        dict: annotation file path
    """
    config = get_config(vars())
    dsa_annotation = qupath_polygon_main(
        config["input_url"],
        config["annotation_name"],
        config["classes_to_include"],
        config["line_colors"],
        config["fill_colors"],
        config["storage_options"],
    )

    annotatation_filepath = save_dsa_annotation(
        dsa_annotation,
        config["output_url"],
        config["image_filename"],
        config["storage_options"],
    )
    return {"dsa_annotation": annotatation_filepath}


def qupath_polygon_main(
    input_url: str,
    annotation_name: str,
    classes_to_include: list,
    line_colors: Optional[dict[str, str]],
    fill_colors: Optional[dict[str, str]],
    storage_options: dict,
):
    """Build DSA annotation json from Qupath polygon geojson

    Args:
        input_url (string): url of Qupath polygon geojson
        annotation_name (string): name of the annotation to be displayed in DSA
        classes_to_include (list): list of classification labels to visualize
        e.g. ["Tumor", "Stroma", ...]
        line_colors (map, optional): line color map with {feature name:rgb
        values}
        fill_colors (map, optional): fill color map with {feature name:rgba
        values}
        storage_options (dict): storage options to pass to read/write functions

    Returns:
        dict: dsa annotation
    """
    regional_file = open(input_url, "r", **storage_options)
    with regional_file.open() as of:
        pixel_clf_polygons = geojson.load(of)

    feature_iter = iter(pixel_clf_polygons)
    if type(pixel_clf_polygons) == geojson.feature.FeatureCollection:
        feature_iter = iter(pixel_clf_polygons.features)

    elements = []
    for polygon in feature_iter:
        props = polygon.properties
        if "classification" not in props:
            continue

        label_name = polygon.properties["classification"]["name"]
        if label_name in classes_to_include:
            element = copy.deepcopy(base_dsa_polygon_element)
            element["label"]["value"] = label_name
            if fill_colors:
                element["fillColor"] = fill_colors[label_name]
            if line_colors:
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
    return get_dsa_annotation(elements, annotation_name)


@timed
@save_metadata
def bitmask_polygon(
    input_map: dict[str, str] = "???",  # type: ignore
    output_url: str = "???",
    image_filename: str = "???",
    annotation_name: str = "???",
    line_colors: Optional[dict[str, str]] = None,
    fill_colors: Optional[dict[str, str]] = None,
    scale_factor: Optional[int] = None,
    storage_options: dict = {},
    local_config: str = "",
):
    """Build DSA annotation json from bitmask PNGs

    Vectorizes and simplifies contours from the bitmask.

    Args:
        input_map (map): map of {label:path_to_bitmask_png}
        output_dir (string): directory to save the DSA compatible annotation
        json
        image_filename (string): name of the image file in DSA e.g. 123.svs
        annotation_name (string): name of the annotation to be displayed in DSA
        line_colors (dict, optional): line color map with {feature name:rgb
        values}
        fill_colors (dict, optional): fill color map with {feature name:rgba
        values}
        scale_factor (int, optional): scale to match the image on DSA. By
        default, 1.
        storage_options (dict): storage options to pass to read/write functions

    Returns:
        dict: annotation file path
    """
    config = get_config(vars())
    dsa_annotation = bitmask_polygon_main(
        config["input_map"],
        config["annotation_name"],
        config["line_colors"],
        config["fill_colors"],
        config["scale_factor"],
        config["storage_options"],
    )
    annotatation_filepath = save_dsa_annotation(
        dsa_annotation,
        config["output_url"],
        config["image_filename"],
        config["storage_options"],
    )
    return {"dsa_annotation": annotatation_filepath}


def bitmask_polygon_main(
    input_map: dict[str, str],
    annotation_name: str,
    line_colors: Optional[dict[str, str]],
    fill_colors: Optional[dict[str, str]],
    scale_factor: Optional[int] = 1,
    storage_options: dict = {},
):
    """Build DSA annotation json from bitmask PNGs

    Vectorizes and simplifies contours from the bitmask.

    Args:
        input (map): map of {label:url_to_bitmask_png}
        annotation_name (string): name of the annotation to be displayed in DSA
        line_colors (dict, optional): line color map with {feature name:rgb
        values}
        fill_colors (dict, optional): fill color map with {feature name:rgba
        values}
        scale_factor (int, optional): scale to match the image on DSA.
        storage_options (dict): storage options to pass to read/write functions

    Returns:
        dict: DSA annotation
    """
    if not check_filepaths_valid(input_map.values(), storage_options):
        raise ValueError("No valid PNG masks found. Exiting..")

    elements = []
    for bitmask_label, bitmask_filepath in input_map.items():
        Image.MAX_IMAGE_PIXELS = 5000000000
        with open(bitmask_filepath, "rb", **storage_options).open() as of:
            annotation = Image.open(of)
            bitmask_np = np.array(annotation)
        simplified_contours = vectorize_np_array_bitmask_by_pixel_value(
            bitmask_np, scale_factor=scale_factor
        )

        for n, contour in enumerate(simplified_contours):
            element = copy.deepcopy(base_dsa_polygon_element)
            label_name = bitmask_label
            element["label"]["value"] = label_name
            if fill_colors:
                element["fillColor"] = fill_colors[label_name]
            if line_colors:
                element["lineColor"] = line_colors[label_name]

            coords = contour.tolist()
            for c in coords:
                c.append(0)
            element["points"] = coords
            elements.append(element)

    return get_dsa_annotation(elements, annotation_name)


@timed
@save_metadata
def heatmap(
    input_url: str = "???",
    output_url: str = "???",
    image_filename: str = "???",
    annotation_name: str = "???",
    column: str = "???",
    tile_size: int = "???",  # type: ignore
    scale_factor: Optional[int] = 1,
    fill_colors: dict[str, str] = {},
    line_colors: dict[str, str] = {},
    storage_options: dict = {},
    local_config: str = "",
):
    """Generate heatmap based on the tile scores

    Creates a heatmap for the given column, using the color palette `viridis`
    to set a fill value
    - the color ranges from purple to yellow, for scores from 0 to 1.

    Args:
        input_url (string): URL to CSV with tile scores
        output_url (string): URL prefix to save the DSA compatible annotation
        json
        image_filename (string): name of the image file in DSA e.g. 123.svs
        annotation_name (string): name of the annotation to be displayed in DSA
        column (string): column to visualize e.g. tile_score
        tile_size (int): size of tiles
        scale_factor (int, optional): scale to match the image on DSA.
        line_colors (dict, optional): line color map with {feature name:rgb
        values}
        fill_colors (dict, optional): fill color map with {feature name:rgba
        values}
        storage_options (dict): storage options to pass to read/write functions
        local_config (string): local config yaml file

    Returns:
        dict: annotation file path. None if error in writing the file.
    """
    config = get_config(vars())
    dsa_annotation = heatmap_main(
        config["input_url"],
        config["annotation_name"],
        config["column"],
        config["tile_size"],
        config["scale_factor"],
        config["fill_colors"],
        config["line_colors"],
        config["storage_options"],
    )
    annotatation_filepath = save_dsa_annotation(
        dsa_annotation,
        config["output_url"],
        config["image_filename"],
        config["storage_options"],
    )
    return {"dsa_annotation": annotatation_filepath}


def heatmap_main(
    input_url: str,
    annotation_name: str,
    column: list[str],
    tile_size: int,
    scale_factor: Optional[int],
    fill_colors: Optional[dict[str, str]],
    line_colors: Optional[dict[str, str]],
    storage_options: dict,
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
        column (list[string]): columns to visualize e.g. tile_score
        tile_size (int): size of tiles
        scale_factor (int, optional): scale to match the image on DSA.
        line_colors (Optional[dict[str,str]]): line color map with {feature name:rgb
        values}
        fill_colors (Optional[dict[str,str]]): fill color map with {feature name:rgba
        values}
        storage_options (dict): storage options to pass to read/write functions

    Returns:
        dict: DSA annotation
    """
    if type(column) == str:
        column = [column]

    df = pd.read_csv(input_url, storage_options=storage_options)
    scaled_tile_size = int(tile_size * int(scale_factor if scale_factor else 1))

    elements = []
    for _, row in df.iterrows():
        element = copy.deepcopy(base_dsa_polygon_element)

        # get label specific color and add to elements
        if len(column) == 1:
            label_value = row[column[0]]
            element["label"]["value"] = str(label_value)
            line_color, fill_color = get_continuous_color(label_value)
            if fill_colors:
                element["fillColor"] = fill_color
            if line_colors:
                element["lineColor"] = line_color
        else:
            label = pd.to_numeric(row[column]).idxmax()
            element["label"]["value"] = str(label)
            if fill_colors:
                element["fillColor"] = fill_colors[label]
            if line_colors:
                element["lineColor"] = line_colors[label]

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

    if len(column) == 1:
        annotation_name = column[0] + "_" + annotation_name

    return get_dsa_annotation(elements, annotation_name)


@timed
@save_metadata
def bmp_polygon(
    input_url: str = "???",
    output_url: str = "???",
    label_map: dict[int, str] = "???",  # type: ignore
    image_filename: str = "???",
    annotation_name: str = "???",
    line_colors: Optional[dict[str, str]] = None,
    fill_colors: Optional[dict[str, str]] = None,
    scale_factor: Optional[int] = 1,
    storage_options: dict = {},
    local_config: str = "",
):
    """Build DSA annotation json from a BMP with multiple labels.

    Vectorizes and simplifies contours per label.

    Args:
        input_url (string): path to bmp file
        output_url (string): url prefix to save the DSA compatible annotation
        json
        label_map (dict[int,str]): map of label number to label name
        image_filename (string): name of the image file in DSA e.g. 123.svs
        annotation_name (string): name of the annotation to be displayed in DSA
        line_colors (dict[str,str], optional): line color map with {feature name:rgb
        values}
        fill_colors (dict[str,str], optional): fill color map with {feature name:rgba
        values}
        scale_factor (int, optional): scale to match image DSA.
        storage_options (dict): storage options to pass to read/write functions

    Returns:
        dict: annotation file path
    """
    config = get_config(vars())
    dsa_annotation = bmp_polygon_main(
        config["input_url"],
        config["label_map"],
        config["annotation_name"],
        config["line_colors"],
        config["scale_factor"],
        config["storage_options"],
    )

    annotatation_filepath = save_dsa_annotation(
        dsa_annotation,
        config["output_url"],
        config["image_filename"],
        config["storage_options"],
    )
    return {"dsa_annotation": annotatation_filepath}


def bmp_polygon_main(
    input_url: str,
    label_map: dict[int, str],
    annotation_name: str,
    line_colors: Optional[dict[str, str]],
    fill_colors: Optional[dict[str, str]],
    scale_factor: Optional[int] = 1,
    storage_options: dict = {},
):
    """Build DSA annotation json from a BMP with multiple labels.

    Vectorizes and simplifies contours per label.

    Args:
        input_url (string): path to bmp file
        label_map (dict[int,str]): map of label number to label name
        annotation_name (string): name of the annotation to be displayed in DSA
        line_colors (dict[str,str], optional): line color map with {feature name:rgb
        values}
        fill_colors (dict[str,str], optional): fill color map with {feature name:rgba
        values}
        scale_factor (int, optional): scale to match image DSA.
        storage_options (dict): storage options to pass to read/write functions

    Returns:
        dict: DSA annotation
    """
    elements = []
    Image.MAX_IMAGE_PIXELS = 5000000000
    with open(input_url, **storage_options).open() as of:
        annotation = Image.open(of)
    arr = np.array(annotation)

    for label_num, label_name in label_map.items():
        simplified_contours = vectorize_np_array_bitmask_by_pixel_value(
            arr, label_num, scale_factor=scale_factor
        )

        for n, contour in enumerate(simplified_contours):
            element = copy.deepcopy(base_dsa_polygon_element)
            element["label"]["value"] = label_name
            if fill_colors:
                element["fillColor"] = fill_colors[label_name]
            if line_colors:
                element["lineColor"] = line_colors[label_name]

            coords = contour.tolist()
            for c in coords:
                c.append(0)
            element["points"] = coords
            elements.append(element)

    return get_dsa_annotation(elements, annotation_name)


if __name__ == "__main__":
    fire.Fire(
        {
            "stardist-polygon": stardist_polygon,
            "stardist-cell": stardist_cell,
            "regional-polygon": regional_polygon,
            "qupath-polygon": qupath_polygon,
            "bitmask-polygon": bitmask_polygon,
            "heatmap": heatmap,
            "bmp-polygon": bmp_polygon,
        }
    )
