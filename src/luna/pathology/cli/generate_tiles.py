# General imports
import itertools
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import fire
import fsspec
import pandas as pd
from dask.distributed import Client, get_client, progress
from fsspec import open  # type: ignore
from loguru import logger
from multimethod import multimethod
from pandera.typing import DataFrame
from tiffslide import TiffSlide

from luna.common.models import SlideSchema, Tile, TileSchema
from luna.common.utils import get_config, save_metadata, timed
from luna.pathology.common.utils import (
    coord_to_address,
    get_full_resolution_generator,
    get_scale_factor_at_magnification,
)


@timed
@save_metadata
def cli(
    slide: str = "???",
    tile_size: int = "???",
    requested_magnification: Optional[int] = None,
    storage_options: dict = {},
    local_config: str = "",
    output_url: str = ".",
) -> dict:
    """Rasterize a slide into smaller tiles, saving tile metadata as rows in a csv file

    Necessary data for the manifest file are:
    address, x_coord, y_coord, xy_extent, tile_size, tile_units

    \b
    Inputs:
        input_slide_image: slide image (virtual slide formats compatible with openslide, .svs, .tif, .scn, ...)
    Outputs:
        slide_tiles
    \b
    Example:
        generate_tiles 10001.svs
            -rts 244 -rmg 10
            -o 10001/tiles
    """
    config = get_config(vars())

    output_filesystem, output_urlpath = fsspec.core.url_to_fs(
        config["output_url"], **config["output_storage_options"]
    )
    slide_id = Path(urlparse(config["slide"]).path).stem
    output_header_file = Path(output_urlpath) / f"{slide_id}.tiles.parquet"

    df = generate_tiles(
        config["slide"],
        config["tile_size"],
        config["storage_options"],
        config["requested_magnification"],
    )
    with output_filesystem.open(output_header_file, "wb") as of:
        print(f"saving to {output_header_file}")
        df.to_parquet(of)

    df.to_parquet(output_header_file)

    properties = {
        "slide_tiles": output_header_file,  # "Tiles" are the metadata that describe them
        "total_tiles": len(df),
        "segment_keys": {"slide_id": str(slide_id)},
    }

    return properties


@multimethod
def generate_tiles(
    slide_manifest: DataFrame[SlideSchema],
    tile_size: int,
    storage_options: dict = {},
    requested_magnification: Optional[int] = None,
) -> pd.DataFrame:
    client = get_client()
    futures = {
        row.id: client.submit(
            generate_tiles, row.url, tile_size, storage_options, requested_magnification
        )
        for row in slide_manifest.itertuples()
    }
    progress(futures)
    results = client.gather(futures)
    for k, v in results.items():
        v["id"] = str(k)
    tiles = pd.concat(results)
    return tiles.merge(slide_manifest, on="id")


@multimethod
def generate_tiles(
    slide_urls: list[str],
    tile_size: int,
    storage_options: dict = {},
    requested_magnification: Optional[int] = None,
) -> pd.DataFrame:
    client = get_client()
    futures = {
        Path(urlparse(slide_url).path).stem: client.submit(
            generate_tiles,
            slide_url,
            tile_size,
            storage_options,
            requested_magnification,
        )
        for slide_url in slide_urls
    }
    progress(futures)
    results = client.gather(futures)
    for k, v in results.items():
        v["id"] = str(k)
    return pd.concat(results)


@multimethod
def generate_tiles(
    slide_url: str,
    tile_size: int,
    storage_options: dict = {},
    requested_magnification: Optional[int] = None,
) -> pd.DataFrame:
    """Rasterize a slide into smaller tiles

    Tiles addresses and arrays are saved as key-value pairs in (tiles.h5),
    and the corresponding manifest/header file (tiles.csv) is also generated

    Necessary data for the manifest file are:
    address, tile_image_file, full_resolution_tile_size, tile_image_size_xy

    Args:
        slide_url (str): slide url
        tile_size (int): size of tiles to use (at the requested magnification)
        requested_magnification (float): Magnification scale at which to perform computation

    Returns:
        DataFrame[TileSchema]: tile manifest
    """
    with open(slide_url, "rb", **storage_options) as f:
        slide = TiffSlide(f)
        logger.info("Slide size = [%s,%s]", slide.dimensions[0], slide.dimensions[1])

        to_mag_scale_factor = get_scale_factor_at_magnification(
            slide, requested_magnification=requested_magnification
        )

        if not to_mag_scale_factor % 1 == 0:
            logger.error(f"Bad magnficiation scale factor = {to_mag_scale_factor}")
            raise ValueError(
                "You chose a combination of requested tile sizes and magnification that resulted in non-integer tile sizes at different scales"
            )

        full_resolution_tile_size = int(tile_size * to_mag_scale_factor)
        logger.info(
            "Normalized magnification scale factor for %sx is %s",
            requested_magnification,
            to_mag_scale_factor,
        )
        logger.info(
            "Requested tile size=%s, tile size at full magnficiation=%s",
            tile_size,
            full_resolution_tile_size,
        )

        # get DeepZoomGenerator, level
        full_generator, full_level = get_full_resolution_generator(
            slide, tile_size=full_resolution_tile_size
        )

    tile_x_count, tile_y_count = full_generator.level_tiles[full_level]
    logger.info(f"tiles x {tile_x_count}, tiles y {tile_y_count}")

    # populate address, coordinates
    tiles = DataFrame[TileSchema](
        [
            Tile(
                address=coord_to_address(address, requested_magnification),
                x_coord=(address[0]) * full_resolution_tile_size,
                y_coord=(address[1]) * full_resolution_tile_size,
                xy_extent=full_resolution_tile_size,
                size=tile_size,
                units="px",
            ).__dict__
            for address in itertools.product(
                range(1, tile_x_count - 1), range(1, tile_y_count - 1)
            )
        ]
    )

    logger.info(f"Number of tiles in raster: {len(tiles)}")
    #    logger.info("Creating lazy tiles")
    #    lazy_tiles = [
    #            [dask.delayed(get_tile_from_slide)(tiles_df(x, y),
    #                                               full_resolution_tile_size,
    #                                               tile_size,
    #                                               slide)
    #             for y in range(1, tile_y_count - 1)]
    #            for x in range(1, tile_x_count - 1)]
    #    sample = lazy_tiles[0][0].compute()
    #
    #    lazy_arrays = da.stack([
    #        da.stack([da.from_delayed(lazy_tile, dtype=sample.dtype, shape=sample.shape)
    #                        for lazy_tile in inner] )
    #        for inner in lazy_tiles
    #        ])
    #    logger.info(f"lazy tiles: {lazy_arrays.shape}")

    return tiles


if __name__ == "__main__":
    Client()
    fire.Fire(cli)
