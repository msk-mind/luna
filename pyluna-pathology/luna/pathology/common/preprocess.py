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
