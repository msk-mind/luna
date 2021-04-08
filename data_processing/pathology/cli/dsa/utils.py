from random import randint

import numpy as np
from skimage import measure
import seaborn as sns

# TODO move to configs...
line_colors_cell = {'Other': "rgb({}, {}, {})".format(0,255,0), "Lymphocyte": "rgb({}, {}, {})".format(255,0,0) }
fill_colors_cell = {'Other': "rgba({}, {}, {}, alpha)".format(0,255,0), "Lymphocyte": "rgba({}, {}, {}, alpha)".format(255,0,0) }

line_colors_regional = {'Stroma': "rgb({}, {}, {})".format(0,191,255), "Tumor": "rgb({}, {}, {})".format(0,255,0), 'Fat': "rgb({}, {}, {})".format(255,255,0), 'Necrosis': "rgb({}, {}, {})".format(255,0,0)}
fill_colors_regional = {'Stroma': "rgba({}, {}, {}, alpha)".format(0,191,255), "Tumor": "rgba({}, {}, {}, alpha)".format(0,255,0), 'Fat': "rgba({}, {}, {}, alpha)".format(255,255,0), 'Necrosis': "rgba({}, {}, {}, alpha)".format(255,0,0)}

line_colors_default = {}
fill_colors_default = {}

# get colors for cells/regions based on discrete categories
# set object_type to "cell" for cell specific labeling (red hot lymphocytes in a forest of green)
# set object_type to "regional" for regional specific labeling (following Kevin's color schema)
def get_color(name, object_type='default', alpha = 100):
    """
    Get colors for cells/regions based on discrete categories.

    :param name: feature name e.g. Stroma, Tumor
    :param object_type: "cell" for cell specific labeling, "regional" for regional specific labeling
        for more information, refer to the config.
    :param alpha: alpha value for the fill color. 100 by default
    :return: RGBA line and fill colors
    """
    if object_type == "cell":
        line_colors = line_colors_cell
        for key, val in fill_colors_cell.items():
            fill_colors_cell[key] = val.replace("alpha", str(alpha))
        fill_colors = fill_colors_cell
    elif object_type == "regional":
        line_colors = line_colors_regional
        for key, val in fill_colors_regional.items():
            fill_colors_regional[key] = val.replace("alpha", str(alpha))
        fill_colors = fill_colors_regional
    else:
        # print("Object type not excpted. Cell and Regional Objects supported.")
        line_colors = line_colors_default
        fill_colors = fill_colors_default

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