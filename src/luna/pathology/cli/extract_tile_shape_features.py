# General imports
import itertools
import json
from enum import Enum
from pathlib import Path
from typing import List, Optional

import fire
import fsspec
import geopandas as gpd
import numpy as np
import pandas as pd
import tiffslide
from dask.distributed import progress
from fsspec import open
from loguru import logger
from pandera.typing import DataFrame
from scipy.stats import entropy, kurtosis
from shapely import box

from luna.common.dask import get_or_create_dask_client
from luna.common.models import LabeledTileSchema, SlideSchema
from luna.common.utils import get_config, save_metadata, timed
from luna.pathology.cli.extract_shape_features import extract_shape_features
from luna.pathology.cli.generate_tile_mask import convert_tiles_to_mask
from luna.pathology.common.utils import resize_array


class StatisticalDescriptors(str, Enum):
    ALL = "All"
    QUANTILES = "Quantiles"
    STATS = "Stats"
    DENSITY = "Density"


STATISTICAL_DESCRIPTOR_PERCENTILES = np.arange(0.1, 1, 0.1)
STATISTICAL_DESCRIPTOR_MAP = {
    StatisticalDescriptors.STATS: ["min", "mean", "median", "max", "sum"],
    StatisticalDescriptors.DENSITY: ["var", "skew", ("kurt", kurtosis)],
    StatisticalDescriptors.QUANTILES: [
        (f"{p:.0%}", lambda x: x.quantile(p))
        for p in STATISTICAL_DESCRIPTOR_PERCENTILES
    ],
}
STATISTICAL_DESCRIPTOR_MAP[StatisticalDescriptors.ALL] = list(
    itertools.chain(*STATISTICAL_DESCRIPTOR_MAP.values())
)


class CellularFeatures(str, Enum):
    ALL = "All"
    NUCLEUS = "Nucleus"
    CELL = "Cell"
    CYTOPLASM = "Cytoplasm"
    MEMBRANE = "Membrane"


class PropertyType(str, Enum):
    ALL = "All"
    GEOMETRIC = "Geometric"
    STAIN = "Stain"


PROPERTY_TYPE_MAP = {
    PropertyType.GEOMETRIC: ["Cell", "Nucleus"],
    PropertyType.STAIN: ["Hematoxylin", "Eosin", "DAB"],
}


@timed
@save_metadata
def cli(
    slide_urlpath: str = "???",
    object_urlpath: str = "???",
    tiles_urlpath: str = "???",
    output_urlpath: str = ".",
    resize_factor: int = 16,
    detection_probability_threshold: Optional[float] = None,
    statistical_descriptors: str = StatisticalDescriptors.ALL,
    cellular_features: str = CellularFeatures.ALL,
    property_type: str = PropertyType.ALL,
    include_smaller_regions: bool = False,
    label_cols: List[str] = None,
    storage_options: dict = {},
    output_storage_options: dict = {},
    local_config: str = "",
):
    """Extracts shape and spatial features (HIF features) from a slide mask.

     Args:
        slide_urlpath (str): URL/path to slide (tiffslide supported formats)
        object_urlpath (str): URL/path to object file (geopandas supported formats)
        tiles_urlpath (str): URL/path to tiles manifest (parquet)
        output_urlpath (str): URL/path to output parquet file
        resize_factor (int): factor to downsample slide image
        detection_probability_threshold (Optional[float]): detection probability threshold
        statistical_descriptors (str): statistical descriptors to calculate. One of All, Quantiles, Stats, or Density
        cellular_features (str): cellular features to include. One of All, Nucleus, Cell, Cytoplasm, and Membrane
        property_type (str): properties to include. One of All, Geometric, or Stain
        include_smaller_regions (bool): include smaller regions in output
        label_cols (List[str]): list of score columns to use for the classification. Tile is classified as the column with the max score
        storage_options (dict): storage options to pass to reading functions
        output_storage_options (dict): storage options to pass to writing functions
        local_config (str): local config yaml file

    Returns:
        dict: output paths and the number of features generated
    """
    config = get_config(vars())

    slide_id = Path(config["slide_urlpath"]).stem

    statistical_descriptors = config["statistical_descriptors"].capitalize()
    cellular_features = config["cellular_features"].capitalize()
    property_type = config["property_type"].capitalize()

    properties = __extract_tile_shape_features(
        config["object_urlpath"],
        config["tiles_urlpath"],
        config["slide_urlpath"],
        config["output_urlpath"],
        config["resize_factor"],
        config["detection_probability_threshold"],
        slide_id,
        statistical_descriptors,
        cellular_features,
        property_type,
        config["include_smaller_regions"],
        config["label_cols"],
        config["storage_options"],
        config["output_storage_options"],
    )
    return properties


def extract_tile_shape_features(
    slide_manifest: DataFrame[SlideSchema],
    output_urlpath: str,
    resize_factor: int = 16,
    detection_probability_threshold: Optional[float] = None,
    statistical_descriptors: StatisticalDescriptors = StatisticalDescriptors.ALL,
    cellular_features: CellularFeatures = CellularFeatures.ALL,
    property_type: PropertyType = PropertyType.ALL,
    include_smaller_regions: bool = False,
    label_cols: List[str] = None,
    storage_options: dict = {},
    output_storage_options: dict = {},
    objects_column="stardist_geojson_url",
    annotation_column="tile_shape_features_url",
    properties: List[str] = [
        "area",
        "convex_area",
        "eccentricity",
        "equivalent_diameter",
        "euler_number",
        "extent",
        "label",
        "major_axis_length",
        "minor_axis_length",
        "perimeter",
        "solidity",
    ],
):
    """Extracts shape and spatial features (HIF features) from a slide mask.

     Args:
        slide_manifest (DataFrame[SlideSchema]): slide manifest from slide_etl
        output_urlpath (str): output URL/path
        resize_factor (int): factor to downsample slide image
        detection_probability_threshold (Optional[float]): detection probability threshold
        statistical_descriptors (str): statistical descriptors to calculate. One of All, Quantiles, Stats, or Density
        cellular_features (str): cellular features to include. One of All, Nucleus, Cell, Cytoplasm, and Membrane
        property_type (str): properties to include. One of All, Geometric, or Stain
        include_smaller_regions (bool): include smaller regions in output
        label_cols (List[str]): list of score columns to use for the classification. Tile is classified as the column with the max score
        storage_options (dict): storage options to pass to reading functions
        output_storage_options (dict): storage options to pass to writing functions
        local_config (str): local config yaml file
        objects_column (str): slide manifest column name with stardist geoJSON URLs
        annotation_column (str): column to add to slide manifest with url to extracted features
        properties (List[str]): properties to extract

    Returns:
        DataFrame[SlideSchema]: slide manifest
    """
    client = get_or_create_dask_client()

    futures = []
    for _, row in slide_manifest.iterrows():
        future = client.submit(
            __extract_tile_shape_features,
            row[objects_column],
            row["tiles_url"],
            row["url"],
            output_urlpath,
            resize_factor,
            detection_probability_threshold,
            row["id"],
            statistical_descriptors,
            cellular_features,
            property_type,
            include_smaller_regions,
            label_cols,
            storage_options,
            output_storage_options,
            properties,
        )
        futures.append(future)

    progress(futures)
    results = client.gather(futures)

    return slide_manifest.assign(
        **{annotation_column: [x["shape_features_url"] for x in results]}
    )


def __extract_tile_shape_features(
    objects_urlpath: str,
    tiles_urlpath: str,
    slide_urlpath: str,
    output_urlpath: str,
    resize_factor: int = 16,
    detection_probability_threshold: Optional[float] = None,
    slide_id: str = "",
    statistical_descriptors: StatisticalDescriptors = StatisticalDescriptors.ALL,
    cellular_features: CellularFeatures = CellularFeatures.ALL,
    property_type: PropertyType = PropertyType.ALL,
    include_smaller_regions: bool = False,
    label_cols: List[str] = None,
    storage_options: dict = {},
    output_storage_options: dict = {},
    properties: List[str] = [
        "area",
        "convex_area",
        "eccentricity",
        "equivalent_diameter",
        "euler_number",
        "extent",
        "label",
        "major_axis_length",
        "minor_axis_length",
        "perimeter",
        "solidity",
    ],
):
    """Extracts shape and spatial features (HIF features) from a slide mask.

     Args:
        objects_urlpath (str): URL/path to object file (geopandas supported formats)
        tiles_urlpath (str): URL/path to tiles manifest (parquet)
        slide_urlpath (str): URL/path to slide (tiffslide supported formats)
        output_urlpath (str): output URL/path
        resize_factor (int): factor to downsample slide image
        detection_probability_threshold (Optional[float]): detection
            probability threshold
        slide_id (str): Slide ID to add to dataframes
        statistical_descriptors (StatisticalDescriptors): statistical descriptors to calculate
        cellular_features (CellularFeatures): cellular features to include
        property_type (PropertyType): properties to include
        include_smaller_regions (bool): include smaller regions
        label_cols (List[str]): list of score columns to use for the classification. Tile is classified as the column with the max score
        storage_options (dict): storage options to pass to reading functions
        output_storage_options (dict): storage options to pass to writing functions
        properties (List[str]): list of whole slide image properties to
            extract. Needs to be parquet compatible (numeric).
    Returns:
        dict: output paths and the number of features generated
    """

    ofs, path = fsspec.core.url_to_fs(
        output_urlpath,
        **output_storage_options,
    )

    output_fpath = Path(path) / "shape_features.parquet"

    if ofs.exists(str(output_fpath)):
        logger.info(
            f"Output file already exist: {ofs.unstrip_protocol(str(output_fpath))}"
        )
        return {}

    with open(tiles_urlpath, **storage_options) as of:
        tiles_df = pd.read_parquet(of)

    with open(objects_urlpath, **storage_options) as of:
        object_gdf = gpd.read_file(of)

    with open(slide_urlpath, **storage_options) as of:
        slide = tiffslide.TiffSlide(of)
        slide_width = slide.dimensions[0]
        slide_height = slide.dimensions[1]

    if label_cols:
        tiles_df["Classification"] = tiles_df[label_cols].idxmax(axis=1)
    LabeledTileSchema.validate(tiles_df.reset_index())

    tile_area = tiles_df.iloc[0].tile_size ** 2

    counts = tiles_df.Classification.value_counts()

    combis = itertools.combinations(counts.index, 2)
    joint_entropy = []
    for i, j in combis:
        ent = {}
        ent["Parent"] = "whole_region"
        ent["Class"] = i
        ent["variable"] = f"Joint Entropy to {j}"
        ent["value"] = entropy(counts[[i, j]], base=2)
        joint_entropy.append(ent)

    entropy_df = pd.DataFrame(joint_entropy)

    shannon_entropy = entropy(counts, base=2)
    entropy_df = entropy_df.append(
        {
            "Parent": "whole_region",
            "Class": "All",
            "variable": "Entropy",
            "value": shannon_entropy,
        },
        ignore_index=True,
    )

    slide_area = counts * tile_area
    slide_area.index.name = "Parent"

    mask, mask_values = convert_tiles_to_mask(
        tiles_df, slide_width, slide_height, "Classification"
    )

    resized_mask = resize_array(mask, resize_factor)
    shape_features_df = extract_shape_features(
        resized_mask, mask_values, include_smaller_regions, properties
    )

    ann_region_polygons = [
        box(
            row.x_coord,
            row.y_coord,
            row.x_coord + row.xy_extent,
            row.y_coord + row.xy_extent,
        )
        for _, row in tiles_df.iterrows()
    ]
    tiles_gdf = gpd.GeoDataFrame(
        data=tiles_df, geometry=ann_region_polygons, crs="EPSG:4326"
    )

    logger.info("Spatially joining tiles and objects")
    gdf = object_gdf.sjoin(tiles_gdf, how="inner", predicate="within")
    if len(gdf) == 0:
        logger.info("No objects found within tiles")
        return None
    try:
        measurement_keys = list(gdf.measurements.iloc[0].keys())
        gdf = gdf.join(gdf.measurements.apply(lambda x: pd.Series(x)))
    except Exception:
        measurements = gdf.measurements.apply(
            lambda x: pd.DataFrame(json.loads(x)).set_index("name").squeeze()
        )
        measurement_keys = list(measurements.columns.values)
        gdf = gdf.join(measurements)
    gdf = gdf.join(gdf.classification.apply(lambda x: pd.Series(x)))
    gdf = gdf.rename(columns={"name": "Class", "Classification": "Parent"})

    gdf.Parent = gdf.Parent.astype("category")
    gdf.Class = gdf.Class.astype("category")

    if detection_probability_threshold:
        gdf = gdf.query(f"`Detection probability` > {detection_probability_threshold}")

    agg_keys = measurement_keys.copy()
    agg_keys.remove("Detection probability")
    logger.info("Calculating object measurement statistics")
    gb = gdf.groupby(by=["Parent", "Class"])[agg_keys]
    agg_funs = STATISTICAL_DESCRIPTOR_MAP[statistical_descriptors]
    agg_df = gb.agg(agg_funs)
    agg_df.columns = [" ".join(col).strip() for col in agg_df.columns.values]
    agg_df["Object Counts"] = gb.size()

    agg_df["Normalized Cell Density"] = agg_df["Object Counts"] / slide_area
    if "Cell: Area µm^2 sum" in agg_df.columns:
        agg_df["Cell Density"] = agg_df["Cell: Area µm^2 sum"] / (slide_area / 4)

    logger.info(
        "Calculating obj count log ratios between all tile label obj classification groups"
    )
    count_col = agg_df.columns.get_loc("Object Counts")
    idx0, idx1 = np.triu_indices(len(agg_df), 1)
    np.seterr(divide="ignore")
    ratio_df = pd.DataFrame(
        data={
            "variable": np.array(
                [
                    "Object Count Log Ratio to " + " ".join(row).strip()
                    for row in agg_df.index.values
                ]
            )[idx1],
            "value": np.log(agg_df.iloc[idx0, count_col].values)
            - np.log(agg_df.iloc[idx1, count_col].values),
        },
        index=agg_df.index[idx0],
    )

    if cellular_features != CellularFeatures.ALL:
        agg_df = agg_df.filter(regex=cellular_features)

    if property_type != PropertyType.ALL:
        property_types = PROPERTY_TYPE_MAP[property_type]
        agg_df = agg_df.filter(regex="|".join(property_types))

    mdf = pd.melt(agg_df.reset_index(), id_vars=["Parent", "Class"]).dropna()
    mdf = pd.concat([mdf, ratio_df.reset_index(), shape_features_df, entropy_df])

    if slide_id:
        mdf.insert(loc=0, column="slide_id", value=slide_id)

    mdf[["Parent", "Class", "variable"]] = mdf[["Parent", "Class", "variable"]].replace(
        r"_", " ", regex=True
    )

    with ofs.open(output_fpath, "wb") as of:
        mdf.to_parquet(of)

    props = {
        "shape_features_url": ofs.unstrip_protocol(str(output_fpath)),
        "num_features": len(mdf),
    }

    logger.info(props)

    return props


def fire_cli():
    fire.Fire(cli)


if __name__ == "__main__":
    fire_cli()
