import numpy as np
from luna.pathology.dsa.utils import (
    get_color,
    get_continuous_color,
    vectorize_np_array_bitmask_by_pixel_value,
)


def test_get_color():
    line_colors = {"tumor": "rgb(255,0,0)"}
    fill_colors = {"tumor": "rgba(255,0,0,100)"}
    lc, fc = get_color("stroma", line_colors, fill_colors)

    assert lc.startswith("rgb(")
    assert fc.startswith("rgba(")


def test_get_continuous_color():
    lc, fc = get_continuous_color(0.5, outline_color="black")
    assert lc == "rgb(0, 0, 0)"

    lc, fc = get_continuous_color(0.5, outline_color="white")
    assert lc == "rgb(255, 255, 255)"


def test_vectorize_np_array_bitmask_by_pixel_value():
    arr = np.array([[1, 1, 1, 0], [1, 1, 0, 0], [1, 1, 1, 0]])

    contours = vectorize_np_array_bitmask_by_pixel_value(arr, label_num=1)

    expected = np.array([[2.0, 0.0], [2.0, 2.0]])
    assert np.array_equal(contours[0], expected)
