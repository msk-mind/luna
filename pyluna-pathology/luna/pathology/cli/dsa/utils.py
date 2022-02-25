from random import randint

import numpy as np
from skimage import measure
import seaborn as sns


def get_color(name, line_colors={}, fill_colors={}, alpha=100):
    """Get colors for cells/regions based on discrete categories.

    Args:
        name (string): feature name e.g. Stroma, Tumor
        line_colors (dict, optional): line color map with {feature name:rgb values}
        fill_colors (dict, optional): fill color map with {feature name:rgba values}
        alpha (int, optional): alpha value for the fill color. 100 by default

    Returns:
        string: RGBA values for line and fill colors
    """
    if name not in line_colors and name not in fill_colors:
        r = randint(0, 255)
        g = randint(0, 255)
        b = randint(0, 255)
        fill_colors[name] = "rgba({}, {}, {}, {})".format(r, g, b, alpha)
        line_colors[name] = "rgb({}, {}, {})".format(r, g, b)
    return line_colors[name], fill_colors[name]


def get_continuous_color(value, outline_color="same_as_fill", alpha=100):
    """Get RGBA line and fill colors for value.

    Use color palette `viridis` to set a fill value - the color ranges from purple to yellow,
     for the values from 0 to 1. This function is used in generating a heatmap.

    Args:
        value (float): continuous value in [0,1]
        outline_color (string, optional): manages the color used to outline the border of the annotation.
            by default, uses the same color as fill_color.
        alpha (int, optional): alpha value for the fill color. 100 by default

    Returns:
        string: RGBA line and fill colors
    """
    c = sns.color_palette("viridis", as_cmap=True)
    r, g, b, a = c(value, bytes=True)

    fill_color = "rgba({}, {}, {}, {})".format(r, g, b, alpha)
    if outline_color == "same_as_fill":
        line_color = "rgb({}, {}, {})".format(r, g, b)
    elif outline_color == "black":
        line_color = "rgb({}, {}, {})".format(0, 0, 0)
    elif outline_color == "white":
        line_color = "rgb({}, {}, {})".format(255, 255, 255)
    else:
        return None, None
    return line_color, fill_color


def vectorize_np_array_bitmask_by_pixel_value(
    bitmask_np, label_num=255, polygon_tolerance=1, contour_level=0.5, scale_factor=1
):
    """Get simplified contours from the bitmask

    Args:
        bitmask_np (np.array): a numpy bitmask
        label_num (int, optional): numeric value to filter the numpy array
        polygon_tolerance (float, optional): Maximum distance from original points of polygon
            to approximated polygonal chain. If tolerance is 0, the original coordinate array is returned.
        contour_level (float, optional): Value along which to find contours in the array.
            0.5 by default
        scale_factor (int, optional): scale to match image. default 1

    Returns:
        list: simplified approximated contours
    """
    mask = np.where(bitmask_np == label_num, 1, 0).astype(np.int8)
    contours = measure.find_contours(mask, level=contour_level)
    simplified_contours = [
        measure.approximate_polygon(c, tolerance=polygon_tolerance) for c in contours
    ]
    for _, contour in enumerate(simplified_contours):
        for coord in contour:
            x = int(round(coord[0]))
            y = int(round(coord[1]))
            # switch coordinates, otherwise gets flipped
            coord[0] = y * scale_factor
            coord[1] = x * scale_factor

    return simplified_contours
