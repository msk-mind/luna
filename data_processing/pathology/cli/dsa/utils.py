from random import randint

import numpy as np
from skimage import measure
import seaborn as sns


def get_color(name, line_colors={}, fill_colors={}, alpha = 100):
    """
    Get colors for cells/regions based on discrete categories.

    :param name: feature name e.g. Stroma, Tumor
    :param line_colors: json with name:rgb values
    :param fill_colors: json with name:rgba values
    :param alpha: alpha value for the fill color. 100 by default
    :return: RGBA line and fill colors
    """
    if name not in line_colors and name not in fill_colors:
        r = randint(0, 255)
        g = randint(0, 255)
        b = randint(0, 255)
        fill_colors[name] = "rgba({}, {}, {}, {})".format(r,g,b, alpha)
        line_colors[name] = "rgb({}, {}, {})".format(r,g,b)
    return line_colors[name], fill_colors[name]


def get_continuous_color(value, outline_color='same_as_fill', alpha = 100):
    """
    Get RGBA line and fill colors for value.

    :param value: continuous variable in [0,1]
    :param outline_color: manages the color used to outline the border of the annotation.
        by default, uses the same color as fill_color.
    :param alpha: alpha value for the fill color. 100 by default
    :return: RGBA line and fill colors
    """
    c = sns.color_palette("viridis",as_cmap=True)
    r,g,b,a = c(value, bytes=True)

    fill_color = "rgba({}, {}, {}, {})".format(r,g,b,alpha)
    if outline_color == 'same_as_fill':
        line_color = "rgb({}, {}, {})".format(r,g,b)
    elif outline_color == 'black':
        line_color = "rgb({}, {}, {})".format(0,0,0)
    elif outline_color == 'white':
        line_color = "rgb({}, {}, {})".format(255,255,255)
    else:
        return None,None
    return line_color, fill_color


def vectorize_np_array_bitmask_by_pixel_value(bitmask_np,
                                              label_num = 255, polygon_tolerance = 1, contour_level = .5):
    """
    Get simplified contours from the bitmask

    :param bitmask_np: numpy bitmask
    :param label_num: (optional) numeric value to filter the numpy array
    :param polygon_tolerance: (optional) tolerance for approximation. 1 by default
    :param contour_level: (optional) contour level. 0.5 by default
    :return: simplified approximated contours
    """
    mask = np.where(bitmask_np==label_num,1,0).astype(np.int8)
    contours = measure.find_contours(mask, level = contour_level)
    simplified_contours = [measure.approximate_polygon(c, tolerance=polygon_tolerance) for c in contours]
    for _, contour in enumerate(simplified_contours):
        for coord in contour:
            x = int(round(coord[0]))
            y = int(round(coord[1]))
            # switch coordinates, otherwise gets flipped
            coord[0] = y
            coord[1] = x

    return simplified_contours