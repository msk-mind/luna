# General imports
from pathlib import Path

import fire
import fsspec
import pandas as pd
from loguru import logger
from pandera.typing import DataFrame

from luna.common.dask import get_or_create_dask_client
from luna.common.models import SlideSchema
from luna.common.runners import runner_provider
from luna.common.utils import get_config, local_cache_urlpath, save_metadata, timed


@timed
@save_metadata
def stardist_simple_cli(
    slide_urlpath: str = "???",
    cell_expansion_size: float = "???",  # type: ignore
    image_type: str = "???",
    output_urlpath: str = ".",
    debug_opts: str = "",
    num_cores: int = 1,
    image: str = "mskmind/qupath-stardist:0.4.3",
    use_singularity: bool = False,
    max_heap_size: str = "64G",
    storage_options: dict = {},
    output_storage_options: dict = {},
    local_config: str = "",
):
    """Run stardist using qupath CLI

    Args:
        input_slide_image (str): path to slide image (virtual slide formats compatible with openslide, .svs, .tif, .scn, ...)
        cell_expansion_size (float): size in pixels to expand cell cytoplasm
        num_cores (int): Number of cores to use for CPU parallelization
        image_type (str): qupath image type (BRIGHTFIELD_H_DAB)
        output_urlpath (str): output url/path
        debug_opts (str): debug options passed as arguments to groovy script
        image (str): docker/singularity image
        use_singularity (bool): use singularity instead of docker
        max_heap_size (str): maximum heap size to pass to java options
        storage_options (dict): storage options to pass to reading functions
        output_storage_options (dict): storage options to pass to writing functions
        local_config (str): local config yaml file

    Returns:
        dict: metadata about function call
    """

    config = get_config(vars())

    return __stardist_simple(
        config["slide_urlpath"],
        config["cell_expansion_size"],
        config["image_type"],
        config["output_urlpath"],
        config["debug_opts"],
        config["num_cores"],
        config["image"],
        config["use_singularity"],
        config["max_heap_size"],
        config["storage_options"],
        config["output_storage_options"],
    )


def stardist_simple(
    slide_manifest: DataFrame[SlideSchema],
    cell_expansion_size: float,
    image_type: str,
    output_urlpath: str,
    debug_opts: str,
    num_cores: int,
    image: str,
    use_singularity: bool,
    max_heap_size: str,
    storage_options: dict,
    output_storage_options: dict,
    annotation_column: str = "stardist_geojson_url",
) -> DataFrame[SlideSchema]:
    """Run stardist using qupath CLI on slides in a slide manifest from
    slide_etl. URIs to resulting GeoJSON will be stored in a specified column
    of the returned slide manifest.

    Args:
        slide_manifest (DataFrame[SlideSchema]): slide manifest from slide_etl
        cell_expansion_size (float): size in pixels to expand cell cytoplasm
        image_type (str): qupath image type (BRIGHTFIELD_H_DAB)
        output_urlpath (str): output url/path
        debug_opts (str): debug options passed as arguments to groovy script
        num_cores (int): Number of cores to use for CPU parallelization
        image (str): docker/singularity image
        use_singularity (bool): use singularity instead of docker
        max_heap_size (str): maximum heap size to pass to java options
        storage_options (dict): storage options to pass to reading functions
        output_storage_options (dict): storage options to pass to writing functions
        annotation_column (str): name of column in resulting slide manifest to store GeoJson URIs

    Returns:
        DataFrame[SlideSchema]: slide manifest
    """

    client = get_or_create_dask_client()

    futures = []
    for row in slide_manifest.itertuples(name="Slide"):
        future = client.submit(
            __stardist_simple,
            row.url,
            cell_expansion_size,
            image_type,
            output_urlpath,
            debug_opts,
            num_cores,
            image,
            use_singularity,
            max_heap_size,
            storage_options,
            output_storage_options,
        )
        futures.append(future)
    results = client.gather(futures)
    return slide_manifest.assign(
        **{annotation_column: [x["geojson_url"] for x in results]}
    )


@local_cache_urlpath(
    file_key_write_mode={"slide_urlpath": "r"},
    dir_key_write_mode={"output_urlpath": "w"},
)
def __stardist_simple(
    slide_urlpath: str,
    cell_expansion_size: float,
    image_type: str,
    output_urlpath: str,
    debug_opts: str,
    num_cores: int,
    image: str,
    use_singularity: bool,
    max_heap_size: str,
    storage_options: dict,
    output_storage_options: dict,
) -> dict:
    """Run stardist using qupath CLI on slides in a slide manifest from
    slide_etl. URIs to resulting GeoJSON will be stored in a specified column
    of the returned slide manifest.

    Args:
        slide_urlpath (str): path to slide image (virtual slide formats compatible with openslide, .svs, .tif, .scn, ...)
        cell_expansion_size (float): size in pixels to expand cell cytoplasm
        num_cores (int): Number of cores to use for CPU parallelization
        image_type (str): qupath image type (BRIGHTFIELD_H_DAB)
        output_urlpath (str): output url/path
        debug_opts (str): debug options passed as arguments to groovy script
        image (str): docker/singularity image
        use_singularity (bool): use singularity instead of docker
        max_heap_size (str): maximum heap size to pass to java options
        storage_options (dict): storage options to pass to reading functions
        output_storage_options (dict): storage options to pass to writing functions

    Returns:
        dict: run metadata
    """
    fs, slide_path = fsspec.core.url_to_fs(slide_urlpath, **storage_options)
    ofs, output_path = fsspec.core.url_to_fs(output_urlpath, **output_storage_options)

    slide_id = Path(slide_urlpath).stem
    output_header_file = Path(output_path) / f"{slide_id}_cell_objects.parquet"
    if ofs.exists(output_header_file):
        logger.info(f"outputs already exist: {output_header_file}")
        return

    if ofs.protocol == "file" and not ofs.exists(output_path):
        ofs.mkdir(output_path)

    runner_type = "DOCKER"
    if use_singularity:
        runner_type = "SINGULARITY"

    slide_filename = Path(slide_path).name
    command = f"echo QuPath script --image /inputs/{slide_filename} --args [cellSize={cell_expansion_size},imageType={image_type},{debug_opts}] /scripts/stardist_simple.groovy"
    logger.info(f"Launching QuPath via {runner_type}:{image} ...")
    logger.info(
        f"\tvolumes={slide_urlpath}:'/inputs/{slide_filename}', {slide_path}:'/output_dir'"
    )
    logger.info(f"\tnano_cpus={int(num_cores * 1e9)}")
    logger.info(f"\timage='{image}'")
    logger.info(f"\tcommand={command}")

    volumes_map = {
        slide_path: f"/inputs/{slide_filename}",
        output_path: "/output_dir",
    }

    runner_config = {
        "image": image,
        "command": command,
        "num_cores": num_cores,
        "max_heap_size": max_heap_size,
        "volumes_map": volumes_map,
    }
    runner = runner_provider.get(runner_type, **runner_config)
    executor = runner.run()
    try:
        for line in executor:
            logger.info(line)
    except TypeError:
        print(executor, "is not iterable")

    stardist_output = Path(output_path) / "cell_detections.tsv"

    df = pd.read_csv(stardist_output, sep="\t")
    df.index = "cell-" + df.index.astype(int).astype(str)
    df.index.rename("cell_id", inplace=True)

    df = df.rename(
        columns={"Centroid X µm": "x_coord", "Centroid Y µm": "y_coord"}
    )  # x,ys follow this convention

    with ofs.open(output_header_file, "wb") as of:
        df.to_parquet(of)

    logger.info("generated cell data:")
    logger.info(df)

    output_geojson_file = Path(output_path) / "cell_detections.geojson"

    properties = {
        "geojson_url": ofs.unstrip_protocol(str(output_geojson_file)),
        "tsv_url": ofs.unstrip_protocol(str(stardist_output)),
        "parquet_url": ofs.unstrip_protocol(str(output_header_file)),
        "spatial": True,
        "total_cells": len(df),
    }

    return properties


@timed
@save_metadata
def stardist_cell_lymphocyte_cli(
    slide_urlpath: str = "???",
    output_urlpath: str = ".",
    num_cores: int = 1,
    use_gpu: bool = False,
    image: str = "mskmind/qupath-stardist:0.4.3",
    use_singularity: bool = False,
    max_heap_size: str = "64G",
    storage_options: dict = {},
    output_storage_options: dict = {},
) -> dict:
    """Run stardist using qupath CLI

    Args:
        slide_urlpath (str): url/path to slide image (virtual slide formats compatible with openslide, .svs, .tif, .scn, ...)
        output_urlpath (str): output url/path
        num_cores (int): Number of cores to use for CPU parallelization
        use_gpu (bool): use GPU
        image (str): docker/singularity image
        use_singularity (bool): use singularity instead of docker
        max_heap_size (str): maximum heap size to pass to java options
        storage_options (dict): storage options to pass to reading functions
        output_storage_options (dict): storage options to pass to writing functions

    Returns:
        dict: run metadata
    """
    config = get_config(vars())
    slide_id = Path(config["slide_urlpath"]).stem
    properties = __stardist_cell_lymphocyte(
        config["slide_urlpath"],
        config["output_urlpath"],
        slide_id,
        config["num_cores"],
        config["use_gpu"],
        config["image"],
        config["use_singularity"],
        config["max_heap_size"],
        config["storage_options"],
        config["output_storage_options"],
    )
    return properties


def stardist_cell_lymphocyte(
    slide_manifest: DataFrame[SlideSchema],
    output_urlpath: str,
    num_cores: int,
    use_gpu: bool = False,
    image: str = "mskmind/qupath-stardist:0.4.3",
    use_singularity: bool = False,
    max_heap_size: str = "64G",
    storage_options: dict = {},
    output_storage_options: dict = {},
    annotation_column: str = "lymphocyte_geojson_url",
) -> DataFrame[SlideSchema]:
    """Run stardist using qupath CLI

    Args:
        slide_manifest (DataFrame[SlideSchema]): slide manifest from slide_etl
        output_urlpath (str): output url/path
        num_cores (int): Number of cores to use for CPU parallelization
        use_gpu (bool): use GPU
        image (str): docker/singularity image
        use_singularity (bool): use singularity instead of docker
        max_heap_size (str): maximum heap size to pass to java options
        storage_options (dict): storage options to pass to reading functions
        output_storage_options (dict): storage options to pass to writing functions
        annotation_column (str): name of column in resulting slide manifest to store GeoJson URIs

    Returns:
        DataFrame[SlideSchema]: slide manifest
    """
    client = get_or_create_dask_client()

    futures = []
    for row in slide_manifest.itertuples(name="Slide"):
        fs, output_path = fsspec.core.url_to_fs(
            output_urlpath, **output_storage_options
        )
        future = client.submit(
            __stardist_cell_lymphocyte,
            row.url,
            fs.unstrip_protocol(str(Path(output_path) / row.id)),
            row.id,
            num_cores,
            use_gpu,
            image,
            use_singularity,
            max_heap_size,
            storage_options,
            output_storage_options,
        )
        futures.append(future)
    results = client.gather(futures)
    return slide_manifest.assign(
        **{annotation_column: [x["geojson_url"] for x in results]}
    )


@local_cache_urlpath(
    file_key_write_mode={"slide_urlpath": "r"},
    dir_key_write_mode={"output_urlpath": "w"},
)
def __stardist_cell_lymphocyte(
    slide_urlpath: str,
    output_urlpath: str,
    slide_id: str,
    num_cores: int,
    use_gpu: bool = False,
    image: str = "mskmind/qupath-stardist:0.4.3",
    use_singularity: bool = False,
    max_heap_size: str = "64G",
    storage_options: dict = {},
    output_storage_options: dict = {},
) -> dict:
    """Run stardist using qupath CLI

    Args:
        slide_urlpath (str): url/path to slide image (virtual slide formats compatible with openslide, .svs, .tif, .scn, ...)
        output_urlpath (str): output url/path
        num_cores (int): Number of cores to use for CPU parallelization
        use_gpu (bool): use GPU
        image (str): docker/singularity image
        use_singularity (bool): use singularity instead of docker
        max_heap_size (str): maximum heap size to pass to java options
        storage_options (dict): storage options to pass to reading functions

    Returns:
        dict: run metadata
    """
    fs, slide_path = fsspec.core.url_to_fs(slide_urlpath, **storage_options)
    ofs, output_path = fsspec.core.url_to_fs(output_urlpath, **output_storage_options)

    output_header_file = Path(output_path) / f"{slide_id}_cell_objects.parquet"
    if ofs.exists(output_header_file):
        logger.info(f"outputs already exist: {output_header_file}")
        return

    if ofs.protocol == "file" and not ofs.exists(output_path):
        ofs.mkdir(output_path)

    qupath_cmd = "QuPath-cpu"
    if use_gpu:
        qupath_cmd = "QuPath-gpu"

    runner_type = "DOCKER"
    if use_singularity:
        runner_type = "SINGULARITY"

    slide_filename = Path(slide_path).name
    command = f"{qupath_cmd} script --image /inputs/{slide_filename} /scripts/stardist_nuclei_and_lymphocytes.groovy"
    logger.info(f"Launching {runner_type} container:")
    logger.info(
        f"\tvolumes={slide_path}:'/inputs/{slide_filename}', {output_path}:'/output_dir'"
    )
    logger.info(f"\tnano_cpus={int(num_cores * 1e9)}")
    logger.info(f"\timage='{image}'")
    logger.info(f"\tcommand={command}")

    volumes_map = {
        slide_path: f"/inputs/{slide_filename}",
        output_path: "/output_dir",
    }

    runner_config = {
        "image": image,
        "command": command,
        "num_cores": num_cores,
        "max_heap_size": max_heap_size,
        "volumes_map": volumes_map,
        "use_gpu": use_gpu,
    }
    runner = runner_provider.get(runner_type, **runner_config)
    executor = runner.run()
    try:
        for line in executor:
            logger.info(line)
    except TypeError:
        print(executor, "is not iterable")

    stardist_output = Path(output_path) / "cell_detections.tsv"

    df = pd.read_csv(stardist_output, sep="\t")
    df.index = "cell-" + df.index.astype(int).astype(str)
    df.index.rename("cell_id", inplace=True)

    df = df.rename(
        columns={"Centroid X µm": "x_coord", "Centroid Y µm": "y_coord"}
    )  # x,ys follow this convention

    with fs.open(output_header_file, "wb") as of:
        df.to_parquet(of)

    logger.info("generated cell data:")
    logger.info(df)

    output_geojson_file = Path(output_path) / "cell_detections.geojson"

    properties = {
        "geojson_url": ofs.unstrip_protocol(str(output_geojson_file)),
        "tsv_url": ofs.unstrip_protocol(str(stardist_output)),
        "parquet_url": ofs.unstrip_protocol(str(output_header_file)),
        "spatial": True,
        "total_cells": len(df),
    }

    return properties


def fire_cli():
    fire.Fire(
        {
            "simple": stardist_simple_cli,
            "cell-lymphocyte": stardist_cell_lymphocyte_cli,
        }
    )


if __name__ == "__main__":
    fire_cli()
