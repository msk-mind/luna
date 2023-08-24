# This is just luna.pathology/common/utils.py

import re
import xml.etree.ElementTree as et
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2  # type: ignore
import fsspec
import h5py  # type: ignore
import numpy as np
import numpy.typing as npt
import pandas as pd
import radiomics  # type: ignore
import seaborn as sns  # type: ignore
import SimpleITK as sitk
from fsspec import open
from loguru import logger
from PIL import Image
from skimage.draw import rectangle_perimeter  # type: ignore
from tiffslide import TiffSlide
from tqdm import tqdm

from luna.common.models import Tile
from luna.common.utils import timed
from luna.pathology.common.deepzoom import DeepZoomGenerator

palette = sns.color_palette("viridis", as_cmap=True)
categorial = sns.color_palette("Set1", 8)
categorical_colors = {}  # type: Dict[str, npt.ArrayLike]


# def get_labelset_keys():
#    """get labelset keys
#
#    Given DATA_CFG, return slideviewer labelsets
#
#    Args:
#        none
#
#    Returns:
#        list: a list of labelset names
#    """
#    cfg = ConfigSet()
#    label_config = cfg.get_value(path=const.DATA_CFG + "::LABEL_SETS")
#    labelsets = [cfg.get_value(path=const.DATA_CFG + "::USE_LABELSET")]
#
#    if cfg.get_value(path=const.DATA_CFG + "::USE_ALL_LABELSETS"):
#        labelsets = list(label_config.keys())
#
#    return labelsets


def get_layer_names(xml_urlpath, storage_options={}):
    """get available layer names

    Finds all possible annotation layer names from a Halo generated xml ROI file

    Args:
        xml_urlpath (str): absolute or relativefile path to input halo XML file. prefix scheme to use alternative filesystems.

    Returns:
        set: Available region names
    """  # Annotations >>
    with open(xml_urlpath, "r", **storage_options) as of:
        e = et.parse(of).getroot()
    e = e.findall("Annotation")
    names = set()

    [names.add(ann.get("Name")) for ann in e]

    return names


def convert_xml_to_mask(
    xml_urlpath: str,
    shape: list,
    annotation_name: str,
    storage_options: dict = {},
) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
    """convert xml to bitmask

    Converts a sparse halo XML annotation file (polygons) to a dense bitmask

    Args:
        xml_urlpath (str): file path to input halo XML file
        shape (list): desired polygon shape
        annotation_name (str): name of annotation

    Returns:
        Optional[Tuple[np.ndarray, Dict[str, Any]]]: annotation bitmask of specified shape
    """

    ret = None
    # Annotations >>
    with open(xml_urlpath, **storage_options) as of:
        e = et.parse(of).getroot()
    e = e.findall("Annotation")
    n_regions = 0
    for ann in e:
        if ann.get("Name") != annotation_name:
            continue

        logger.debug(f"Found region {ann.get('Name')}")

        board_pos = np.zeros(shape, dtype=np.uint8)
        board_neg = np.zeros(shape, dtype=np.uint8)

        regions = ann.findall("Regions")
        assert len(regions) == 1

        rs = regions[0].findall("Region")

        for i, r in enumerate(rs):
            negative_flag = int(r.get("NegativeROA"))
            assert negative_flag == 0 or negative_flag == 1
            negative_flag = bool(negative_flag)

            vs = r.findall("Vertices")[0]
            vs = vs.findall("V")
            vs.append(vs[0])  # last dot should be linked to the first dot

            plist = list()
            for v in vs:
                x, y = int(v.get("X").split(".")[0]), int(v.get("Y").split(".")[0])
                plist.append((x, y))

            if negative_flag:
                board_neg = cv2.drawContours(
                    board_neg, [np.array(plist, dtype=np.int32)], -1, [0, 0, 0], -1
                )
            else:
                board_pos = cv2.drawContours(
                    board_pos,
                    [np.array(plist, dtype=np.int32)],
                    contourIdx=-1,
                    color=[255, 0, 0],
                    thickness=-1,
                )
            n_regions += 1

        ret = (board_pos > 0) * (board_neg == 0)

    if ret.any():
        mask = ret.astype(np.uint8)

        properties = {
            "n_regions": n_regions,
            "n_positive_pixels": np.where(mask > 0, 1, 0).sum(),
        }
        return mask, properties
    return None


def convert_halo_xml_to_roi(xml_fn: str) -> Optional[Tuple[List, List]]:
    """get roi from halo XML file

    Read the rectangle ROI of a halo XML annotation file

    Args:
        xml_fn: file path to input halo XML file

    Returns:
        Tuple[list, list]: returns a tuple of x, y coordinates of the recangular roi

    """

    ylist = list()
    xlist = list()

    print("Converting to ROI:", xml_fn)
    e = et.parse(xml_fn).getroot()
    for ann in e.findall("Annotation"):
        regions = ann.findall("Regions")[0]
        if len(regions) == 0:
            continue

        if not regions[0].get("Type") == "Rectangle":
            continue

        for i, r in enumerate(regions):
            vs = r.findall("Vertices")[0]
            vs = vs.findall("V")
            for v in vs:
                y, x = int(v.get("Y").split(".")[0]), int(v.get("X").split(".")[0])
                ylist.append(y)
                xlist.append(x)

    if xlist == [] or ylist == []:
        logger.warning("No Rectangle found, returning None!")
        return None

    if min(xlist) < 0:
        logger.warning("Somehow a negative x rectangle coordinate!")
        xlist = [0, max(xlist)]
    if min(ylist) < 0:
        logger.warning("Somehow a negative y rectangle coordinate!")
        ylist = [0, max(ylist)]

    return xlist, ylist


def get_stain_vectors_macenko(sample: np.ndarray) -> np.ndarray:
    """get_stain_vectors

    Uses the staintools MacenkoStainExtractor to extract stain vectors

    Args:
        sample (np.ndarray): input patch
    Returns:
        np.ndarray: the stain matrix

    """
    from staintools.stain_extraction.macenko_stain_extractor import (
        MacenkoStainExtractor,  # type: ignore
    )

    extractor = MacenkoStainExtractor()
    vectors = extractor.get_stain_matrix(sample)
    return vectors


def pull_stain_channel(
    patch: np.ndarray, vectors: np.ndarray, channel: Optional[int] = None
) -> np.ndarray:
    """pull stain channel

    adds 'stain channel' to the image patch

    Args:
        patch (np.ndarray): input image patch
        vectors (np.ndarray): stain vectors
        channel (int): stain channel

    Returns:
        np.ndarray: the input image patch with an added stain channel
    """

    from staintools.miscellaneous.get_concentrations import (
        get_concentrations,  # type: ignore
    )

    tile_concentrations = get_concentrations(patch, vectors)
    identity = np.array([[1, 0, 0], [0, 1, 0]])
    tmp = 255 * (1 - np.exp(-1 * np.dot(tile_concentrations, identity)))
    tmp = tmp.reshape(patch.shape).astype(np.uint8)
    if channel is not None:
        return tmp[:, :, channel]
    else:
        return tmp


def extract_patch_texture_features(
    image_patch, mask_patch, stain_vectors, stain_channel, plot=False
) -> Optional[Dict[str, np.ndarray]]:
    """extact patch texture features

    Runs patch-wise extraction from an image_patch, mask_patch pair given a stain
    vector and stain channel.

    Args:
        image_patch (np.ndarray): input image patch
        mask_patch (np.ndarray): input image mask
        stain_vectors (np.ndarray): stain vectors extacted from the image patch
        stain_channel (int): stain channel
        plot (Optional, bool): unused?

    Returns:
        Optional[Dict[str, np.ndarray]]: texture features from image patch

    """

    # logging.getLogger("radiomics.featureextractor").setLevel(logging.WARNING)
    if not (len(np.unique(mask_patch)) > 1 and np.count_nonzero(mask_patch) > 1):
        return None

    output_dict = {}  # type: Dict[str, Any]

    stain_patch = pull_stain_channel(image_patch, stain_vectors, channel=stain_channel)

    original_pixels = stain_patch.astype(np.uint8)[
        np.where(mask_patch.astype(np.bool_))
    ].flatten()
    original_pixels_valid = original_pixels[original_pixels > 0]
    output_dict["original_pixels"] = original_pixels_valid

    extractor = radiomics.featureextractor.RadiomicsFeatureExtractor(binWidth=16)
    extractor.disableAllFeatures()
    extractor.enableImageTypeByName("Original")
    extractor.enableFeatureClassByName("glcm")
    # extractor.enableFeatureByName('original_glcm_MCC', enable=False)

    sitk_image = sitk.GetImageFromArray(stain_patch.astype(np.uint8))
    sitk_mask = sitk.GetImageFromArray(mask_patch.astype(np.uint8))

    try:
        bbox, _ = radiomics.imageoperations.checkMask(sitk_image, sitk_mask)
    except Exception as exc:
        logger.warning(f"Skipping this patch, mask pair due to '{exc}'")
        return None
    else:
        # cimg, cmas = radiomics.imageoperations.cropToTumorMask(sitk_image, sitk_mask, bbox)

        fts = extractor.execute(sitk_image, sitk_mask, voxelBased=True)

        for key in fts.keys():
            if "original_glcm" not in key:
                continue

            stainomics_patch = sitk.GetArrayFromImage(fts[key]).astype(np.float32)
            stainomics_nonzero = stainomics_patch[stainomics_patch != 0].flatten()
            stainomics_valid = stainomics_nonzero[~np.isnan(stainomics_nonzero)]

            output_dict[key] = stainomics_valid

        return output_dict


def get_array_from_tile(
    tile: Tile,
    slide: TiffSlide,
    size: Optional[int] = None,
):
    x, y, extent = tile.x_coord, tile.y_coord, tile.xy_extent
    if size is None:
        resize_size = (tile.tile_size, tile.tile_size)
    else:
        resize_size = (size, size)
    arr = np.array(
        slide.read_region((x, y), 0, (extent, extent)).resize(
            resize_size, Image.NEAREST
        )
    )[:, :, :3]
    return arr


def get_tile_from_slide(
    address: Tuple[int, int],
    full_resolution_tile_size: int,
    tile_size: int,
    slide: TiffSlide,
    resize_size: Optional[int] = None,
):
    x, y = (
        address[0] * full_resolution_tile_size,
        address[1] * full_resolution_tile_size,
    )
    if resize_size is None:
        resize_size = (tile_size, tile_size)
    else:
        resize_size = (resize_size, resize_size)
    tile = np.array(
        slide.read_region(
            (x, y), 0, (full_resolution_tile_size, full_resolution_tile_size)
        ).resize(resize_size, Image.NEAREST)
    )[:, :, :3]
    return tile


def resize_array(
    arr: np.ndarray,
    factor: int,
    resample: Image.Resampling = Image.Resampling.NEAREST,
):
    image = Image.fromarray(arr)
    x, y = image.size
    new_x, new_y = int(x / factor), int(y / factor)
    image = image.resize((new_x, new_y), resample)
    return np.array(image)


def get_tile_arrays(
    indices: List[int],
    input_slide_urlpath: str,
    tile_size: int,
    storage_options: dict = {},
) -> List[Tuple[int, np.ndarray]]:
    """
    Get tile arrays for the tile indices

    Args:
        indices (List[int]): list of integers to return as tiles
        input_slide_image (str): path to WSI
        tile_size (int): width, height of generated tile

    Returns:
        a list of tuples (index, tile array) for given indices
    """
    full_generator, full_level = get_full_resolution_generator(
        input_slide_urlpath, tile_size=tile_size, storage_options=storage_options
    )
    return [
        (
            index,
            np.array(
                full_generator.get_tile(
                    full_level, address_to_coord(str(index))
                ).resize((tile_size, tile_size))
            ),
        )
        for index in indices
    ]


def get_tile_array(row: pd.DataFrame, storage_options: dict = {}) -> np.ndarray:
    """
    Returns a tile image as a numpy array.

    Args:
        row (pd.DataFrame): row with address and tile_image_file columns
    """
    fs, path = fsspec.core.url_to_fs(row.tile_store, **storage_options)
    cache_fs = fsspec.filesystem("filecache", fs=fs)
    with cache_fs.open(path, "rb", **storage_options) as of:
        with h5py.File(of, "r") as hf:
            tile = np.array(hf[row.name])
            return tile


# USED -> utils
def coord_to_address(s: Tuple[int, int], magnification: Optional[int]) -> str:
    """converts coordinate to address

    Args:
        s (tuple[int, int]): coordinate consisting of an (x, y) tuple
        magnification (int): magnification factor

    Returns:
        str: a string consisting of an x_y_z address
    """

    x = s[0]
    y = s[1]
    address = f"x{x}_y{y}"
    if magnification:
        address += f"_z{magnification}"
    return address


# USED -> utils
def address_to_coord(s: str) -> Optional[Tuple[int, int]]:
    """converts address into coordinates

    Args:
        s (str): a string consisting of an x_y_z address

    Returns:
        Tuple[int, int]: a tuple consisting of an x, y pair
    """
    s = str(s)
    p = re.compile(r"x(\d+)_y(\d+)", re.IGNORECASE)
    m = p.match(s)
    if m:
        x = int(m.group(1))
        y = int(m.group(2))
        return (x, y)
    return None


@timed
def get_downscaled_thumbnail(
    slide: TiffSlide, scale_factor: Union[int, float]
) -> np.ndarray:
    """get downscaled thumbnail

    yields a thumbnail image of a whole slide rescaled by a specified scale factor

    Args:
        slide (TiffSlide): slide object
        scale_factor (int): integer scaling factor to resize the whole slide by

    Returns:
        np.ndarray: downsized whole slie thumbnail
    """
    new_width = slide.dimensions[0] // scale_factor
    new_height = slide.dimensions[1] // scale_factor
    img = slide.get_thumbnail((int(new_width), int(new_height)))
    return np.array(img)


def get_full_resolution_generator(
    slide_urlpath: str,
    tile_size: int,
    storage_options: dict = {},
) -> Tuple[DeepZoomGenerator, int]:
    """Return MinimalComputeAperioDZGenerator and generator level

    Args:
        slide_urlpath (str): slide urlpath

    Returns:
        Tuple[MinimalComputeAperioDZGenerator, int]
    """
    generator = DeepZoomGenerator(
        slide_urlpath,
        overlap=0,
        tile_size=tile_size,
        limit_bounds=False,
        storage_options=storage_options,
    )

    generator_level = generator.level_count - 1
    # assert generator.level_dimensions[generator_level] == slide.dimensions
    return generator, generator_level


# USED -> utils
def get_scale_factor_at_magnification(
    slide: TiffSlide, requested_magnification: Optional[int]
) -> float:
    """get scale factor at magnification

    Return a scale factor if slide scanned magnification and
    requested magnification are different.

    Args:
        slide (TiffSlide): slide object
        requested_magnification (Optional[int]): requested magnification

    Returns:
        int: scale factor required to achieve requested magnification
    """
    # First convert to float to handle true integers encoded as string floats (e.g. '20.000')
    mag_value = float(slide.properties["aperio.AppMag"])

    # Then convert to integer
    scanned_magnification = int(mag_value)

    # # Make sure we don't have non-integer magnifications
    if not int(mag_value) == mag_value:
        raise RuntimeError(
            "Can't handle slides scanned at non-integer magnficiations! (yet)"
        )

    # Verify magnification valid
    scale_factor = 1.0
    if requested_magnification and scanned_magnification != requested_magnification:
        if scanned_magnification < requested_magnification:
            raise ValueError(
                f"Expected magnification <={scanned_magnification} but got {requested_magnification}"
            )
        elif (scanned_magnification % requested_magnification) == 0:
            scale_factor = scanned_magnification // requested_magnification
        else:
            logger.warning("Scale factor is not an integer, be careful!")
            scale_factor = scanned_magnification / requested_magnification

    return scale_factor


def visualize_tiling_scores(
    df: pd.DataFrame,
    thumbnail_img: np.ndarray,
    scale_factor: float,
    score_type_to_visualize: str,
    normalize=True,
) -> np.ndarray:
    """visualize tile scores

    draws colored boxes around tiles to indicate the value of the score

    Args:
        df (pd.DataFrame): input dataframe
        thumbnail_img (np.ndarray): input tile
        tile_size (int): tile width/length
        score_type_to_visualize (str): column name from data frame

    Returns:
        np.ndarray: new thumbnail image with boxes around tiles passing indicating the
        value of the score
    """

    assert isinstance(thumbnail_img, np.ndarray)

    if normalize and df[score_type_to_visualize].dtype.kind in "biuf":
        df[score_type_to_visualize] = (
            df[score_type_to_visualize] - np.min(df[score_type_to_visualize])
        ) / np.ptp(df[score_type_to_visualize])

    for _, row in tqdm(df.iterrows(), total=len(df)):
        if "regional_label" in row and pd.isna(row.regional_label):
            continue

        start = (
            row.y_coord / scale_factor,
            row.x_coord / scale_factor,
        )  # flip because OpenSlide uses (column, row), but skimage, uses (row, column)

        rr, cc = rectangle_perimeter(
            start=start,
            extent=(row.xy_extent / scale_factor, row.xy_extent / scale_factor),
            shape=thumbnail_img.shape,
        )

        # set color based on intensity of value instead of black border (1)
        score = row[score_type_to_visualize]

        thumbnail_img[rr, cc] = get_tile_color(score)

    return thumbnail_img


def get_tile_color(score: Union[str, float]) -> Optional[npt.ArrayLike]:
    """get tile color

    uses deafult color palette to return color of tile based on score

    Args:
        score (Union[str, float]): a value between [0,1] such as the
            Otsu threshold, puple score, a model output, etc.
    Returns:
        Union[float, None]: returns the color is the input is of valid type
            else None

    """
    # categorical
    if isinstance(score, str):
        if score in categorical_colors:
            return categorical_colors[score]
        else:
            tile_color = 255 * np.array(categorial[len(categorical_colors.keys())])
            categorical_colors[score] = tile_color
            return tile_color

    # float, expected to be value from [0,1]
    elif isinstance(score, float) and score <= 1.0 and score >= 0.0:
        tile_color = np.array([int(255 * i) for i in palette(score)[:3]])
        return tile_color

    else:
        print("Invalid Score Type")
        return None
