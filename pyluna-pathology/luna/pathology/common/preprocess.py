"""
@author: aukermaa@mskcc.org
@author: pateld6@mskcc.org
@author: rosed2@mskcc.org

Various utility and processing methods for pathology
"""

import os, itertools, logging, re

from typing import Union, Tuple, List, Dict
import numpy  as np
import pandas as pd
import seaborn as sns

import json 
from pathlib import Path

from PIL import Image
from pyarrow.parquet import read_table

import openslide
from openslide.deepzoom import DeepZoomGenerator

from skimage.color   import rgb2gray
from skimage.filters import threshold_otsu
from skimage.draw import rectangle_perimeter, rectangle

import requests
import importlib

from shapely.geometry import shape, Point, Polygon

from luna.common.DataStore import DataStore_v2
from random import randint
import torch

logger = logging.getLogger(__name__)


palette = sns.color_palette("viridis",as_cmap=True)
categorial = sns.color_palette("Set1", 8)
categorical_colors = {}



# USED -> utils
def array_to_slide(arr: np.ndarray) -> openslide.OpenSlide:
    """converts a numpy array to a openslide.OpenSlide object

    Args:
        arr (np.ndarray): input image array
    
    Returns:
        openslide.OpenSlide: a slide object from openslide    
    """

    assert isinstance(arr, np.ndarray)
    slide = openslide.ImageSlide(Image.fromarray(arr))
    return slide



def build_shapely_polygons_from_geojson(annotation_geojson:Dict[str, any])-> Tuple[list,
        list]:
    """Build shapely polygons from geojson

    builds a list of shapely polygons and their cooresponding label from a geojson object

    Args:
        annotation_geojson (dict[str, any]): input annotation geoJSON object
    
    Returns:
        Tuple[list, list]: a tuple consisting of polygon and annotation label lists
    """
    annotation_polygons = []
    annotation_labels = []
    # print(len(annotation_geojson['features']))

    for feature in annotation_geojson['features']:

        class_name = feature['properties']['label_name']

        ring_list = feature['geometry']['coordinates']

        # polygon with no holes
        if len(ring_list) == 1:
            annotation_polygon = Polygon(ring_list[0])
        else:
            # this is a ring with interior holes
            annotation_polygon = Polygon(ring_list[0], holes=ring_list[1:])
        
        # check all coord lists are valid

        if annotation_polygon.is_valid:
            annotation_polygons.append(annotation_polygon)
            annotation_labels.append(class_name)


    return annotation_polygons,annotation_labels


def get_regional_labels(address_raster:list, annotation_polygons:list,
        annotation_labels:list, full_generator:DeepZoomGenerator, full_level:int)->list:
    """get regional labels

    Returns annotation labels for tiles that contain annotations
    If the tile doesn't contain annotation, set label to None.

    Args:
        address_raster (list): raster coordinates for tiles 
        annotation_polygons (list): list of shapely Polygon objects
        annotation_labels (list): list of annotation label names
        full_generator (DeepZoomGenerator): whole slide full resolution generator
        param full_level (int): full res level for full_generator
    
    Returns:
        list: list of annotation labels for each polygon 
    """
    regional_label_results = []

    for address in address_raster:
        tile_contains_annotation = False
        tile,_,tile_size = full_generator.get_tile_coordinates(full_level, address)

        tile_x, tile_y = tile
        tile_size_x, tile_size_y = tile_size

        tile_polygon = Polygon([
            (tile_x,               tile_y),
            (tile_x,               tile_y+tile_size_y),
            (tile_x+tile_size_x,   tile_size_y + tile_y),
            (tile_x + tile_size_x, tile_y),
            ])

        for annotation_polygon, annotation_label in zip(annotation_polygons,annotation_labels):
            if annotation_polygon.contains(tile_polygon):
                tile_contains_annotation = True
                regional_label_results.append (annotation_label)
                break
        if not tile_contains_annotation:
            regional_label_results.append (None)

    return regional_label_results


def get_regional_labels_from_table():
    """ TODO: Fix """
    # get pathology annotations for slide only if valid parameters
    if project_id != None and project_id != "" and labelset != None and labelset != "":
        # from get_pathology_annotations
        regional_annotation_table = read_table(annotation_table_path, columns=["geojson_path"],
                                               filters=[('slide_id','=',f'{image_id}'),
                                                        ('user', '=', 'CONCAT'),
                                                        ('labelset', '=', f'{labelset.upper()}')]) \
                                    .to_pandas()
        geojson_path = regional_annotation_table['geojson_path'][0]
        if geojson_path:
            # geojson is saved as a string
            with open(geojson_path) as geojson_file:
                annotation_geojson = json.loads(json.load(geojson_file))

            annotation_polygons, annotation_labels = build_shapely_polygons_from_geojson(annotation_geojson)
            df.loc[:, "regional_label"] = get_regional_labels (df['coordinates'], annotation_polygons, annotation_labels, full_generator, full_level)

