# General imports
import itertools
from pathlib import Path
from typing import Optional

import fire
import fsspec
import pandas as pd
from loguru import logger
from pandera.typing import DataFrame
from tiffslide import TiffSlide

from luna.common.dask import configure_dask_client, get_or_create_dask_client
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
    slide_urlpath: str = "???",
    tile_size: int = "???",  # type: ignore
    requested_magnification: Optional[int] = None,
    storage_options: dict = {},
    output_storage_options: dict = {},
    dask_options: dict = {},
    local_config: str = "",
    output_urlpath: str = ".",
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

    configure_dask_client(**config["dask_options"])

    properties = __generate_tiles(
        config["slide_urlpath"],
        config["tile_size"],
        config["output_urlpath"],
        config["requested_magnification"],
        config["storage_options"],
        config["output_storage_options"],
    )

    return properties


def generate_tiles(
    slide_manifest: DataFrame[SlideSchema],
    tile_size: int,
    output_urlpath: str,
    requested_magnification: Optional[int] = None,
    storage_options: dict = {},
    output_storage_options: dict = {},
) -> pd.DataFrame:
    client = get_or_create_dask_client()

    futures = []
    for slide in slide_manifest.itertuples(name="Slide"):
        future = client.submit(
            __generate_tiles,
            slide.url,
            tile_size,
            output_urlpath,
            requested_magnification,
            storage_options,
            output_storage_options,
        )
        futures.append(future)
    results = client.gather(futures)
    for idx, result in enumerate(results):
        slide_manifest.at[idx, "tiles_url"] = result["tiles_url"]

    return slide_manifest


def __generate_tiles(
    slide_urlpath: str,
    tile_size: int,
    output_urlpath: str,
    requested_magnification: Optional[int] = None,
    storage_options: dict = {},
    output_storage_options: dict = {},
) -> dict:
    """Rasterize a slide into smaller tiles

    Tiles addresses and arrays are saved as key-value pairs in (tiles.h5),
    and the corresponding manifest/header file (tiles.csv) is also generated

    Necessary data for the manifest file are:
    address, tile_image_file, full_resolution_tile_size, tile_image_size_xy

    Args:
        slide_urlpath (str): slide url/path
        tile_size (int): size of tiles to use (at the requested magnification)
        requested_magnification (float): Magnification scale at which to perform computation

    Returns:
        DataFrame[TileSchema]: tile manifest
    """
    slide_id = Path(slide_urlpath).stem
    ofs, output_path = fsspec.core.url_to_fs(output_urlpath, **output_storage_options)
    output_file = str(Path(output_path) / f"{slide_id}.tiles.parquet")
    if ofs.exists(output_file):
        logger.info("Output file exists: {ofs.unstrip_protocol(output_file)}")
        return

    with fsspec.open(slide_urlpath, "rb", **storage_options) as f:
        slide = TiffSlide(f)
        logger.info(f"Slide size = [{slide.dimensions[0]},{slide.dimensions[1]}]")

        to_mag_scale_factor = get_scale_factor_at_magnification(
            slide, requested_magnification=requested_magnification
        )

        if not to_mag_scale_factor % 1 == 0:
            logger.error(f"Bad magnfication scale factor = {to_mag_scale_factor}")
            raise ValueError(
                "You chose a combination of requested tile sizes and magnification that resulted in non-integer tile sizes at different scales"
            )

        full_resolution_tile_size = int(tile_size * to_mag_scale_factor)
        logger.info(
            f"Normalized magnification scale factor for {requested_magnification}x is {to_mag_scale_factor}",
        )
        logger.info(
            f"Requested tile size={tile_size}, tile size at full magnification={full_resolution_tile_size}"
        )

    # get DeepZoomGenerator, level
    full_generator, full_level = get_full_resolution_generator(
        slide_urlpath,
        tile_size=full_resolution_tile_size,
        storage_options=storage_options,
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
                tile_size=tile_size,
                tile_units="px",
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

    with ofs.open(output_file, mode="wb") as of:
        tiles.to_parquet(of)

    properties = {
        "tiles_url": ofs.unstrip_protocol(
            output_file
        ),  # "Tiles" are the metadata that describe them
        "total_tiles": len(tiles),
    }

    return properties


def fire_cli():
    fire.Fire(cli)


if __name__ == "__main__":
    fire_cli()
