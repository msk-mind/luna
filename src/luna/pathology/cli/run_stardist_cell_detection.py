# General imports
from pathlib import Path

import fire
import fsspec
import pandas as pd
from loguru import logger

from luna.common.runners import runner_provider
from luna.common.utils import get_config, local_cache_urlpath, save_metadata, timed


@timed
@save_metadata
def stardist_simple(
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
        pd.DataFrame: metadata about function call
    """

    config = get_config(vars())
    fs, output_path = fsspec.core.url_to_fs(
        config["output_urlpath"], **config["output_storage_options"]
    )
    slide_id = Path(config["slide_urlpath"]).stem
    output_header_file = Path(output_path) / f"{slide_id}_cell_objects.parquet"
    if fs.exists(output_header_file):
        logger.info(f"outputs already exist: {config['output_urlpath']}")
        return

    df = stardist_simple_main(
        config["slide_urlpath"],
        config["cell_expansion_size"],
        config["image_type"],
        config["output_urlpath"],
        config["debug_opts"],
        config["num_cores"],
        config["image"],
        config["use_singularity"],
        config['max_heap_size'],
        config["storage_options"],
        config["output_storage_options"],
    )

    with fs.open(output_header_file, "wb") as of:
        df.to_parquet(of)

    logger.info("generated cell data:")
    logger.info(df)

    output_geojson_file = Path(output_path) / "cell_detections.geojson"

    properties = {
        "cell_objects": str(output_header_file),
        "feature_data": str(output_header_file),
        "geojson_features": str(output_geojson_file),
        "spatial": True,
        "total_cells": len(df),
        "segment_keys": {"slide_id": slide_id},
    }

    return properties


@local_cache_urlpath(
    file_key_write_mode={"slide_urlpath": "r"},
    dir_key_write_mode={"output_urlpath": "w"},
)
def stardist_simple_main(
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
) -> pd.DataFrame:
    """Run stardist using qupath CLI

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
        pd.DataFrame: cell detections
    """
    fs, slide_path = fsspec.core.url_to_fs(slide_urlpath, **storage_options)
    ofs, output_path = fsspec.core.url_to_fs(output_urlpath, **output_storage_options)

    if ofs.protocol == 'file' and not ofs.exists(output_path):
        ofs.mkdir(output_path)

    runner_type = "DOCKER"
    if use_singularity:
        runner_type = "SINGULARITY"

    slide_filename = Path(slide_path).name
    command = f"QuPath script --image /inputs/{slide_filename} --args [cellSize={cell_expansion_size},imageType={image_type},{debug_opts}] /scripts/stardist_simple.groovy"
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
    for line in executor:
        print(line)

    stardist_output = Path(output_path) / "cell_detections.tsv"

    df = pd.read_csv(stardist_output, sep="\t")
    df.index = "cell-" + df.index.astype(int).astype(str)
    df.index.rename("cell_id", inplace=True)

    df = df.rename(
        columns={"Centroid X µm": "x_coord", "Centroid Y µm": "y_coord"}
    )  # x,ys follow this convention

    return df


@timed
@save_metadata
def stardist_cell_lymphocyte(
    slide_urlpath: str = "???",
    output_urlpath: str = ".",
    num_cores: int = 1,
    use_gpu: bool = False,
    image: str = "mskmind/qupath-stardist:0.4.3",
    use_singularity: bool = False,
    max_heap_size: str = "64G",
    storage_options: dict = {},
    output_storage_options: dict = {},
):
    """Run stardist using qupath CLI

    Args:
        slide_urlpath (str): url/path to slide image (virtual slide formats compatible with openslide, .svs, .tif, .scn, ...)
        output_urlpath (str): output url/path
        num_cores (int): Number of cores to use for CPU parallelization
        use_gpu (bool): use GPU
        use_singularity (bool): use singularity instead of docker
        max_heap_size (str): maximum heap size to pass to java options
        storage_options (dict): storage options to pass to reading functions
        output_storage_options (dict): storage options to pass to writing functions

    Returns:
        pd.DataFrame: cell detections
    """
    config = get_config(vars())

    fs, output_path = fsspec.core.url_to_fs(
        config["output_urlpath"], **config["output_storage_options"]
    )
    slide_id = Path(config["slide_urlpath"]).stem
    output_header_file = Path(output_path) / f"{slide_id}_cell_objects.parquet"
    if fs.exists(output_header_file):
        logger.info(f"outputs already exist: {config['output_urlpath']}")
        return

    df = stardist_cell_lymphocyte_main(
        config["slide_urlpath"],
        config["output_urlpath"],
        config["num_cores"],
        config["use_gpu"],
        config["image"],
        config["use_singularity"],
        config["max_heap_size"],
        config["storage_options"],
        config["output_storage_options"],
    )

    with fs.open(output_header_file, "wb") as of:
        df.to_parquet(of)

    logger.info("generated cell data:")
    logger.info(df)

    output_geojson_file = Path(output_path) / "cell_detections.geojson"

    properties = {
        "cell_objects": str(output_header_file),
        "feature_data": str(output_header_file),
        "geojson_features": str(output_geojson_file),
        "spatial": True,
        "total_cells": len(df),
        "segment_keys": {"slide_id": slide_id},
    }

    return properties


@local_cache_urlpath(
    file_key_write_mode={"slide_urlpath": "r"},
    dir_key_write_mode={"output_urlpath": "w"},
)
def stardist_cell_lymphocyte_main(
    slide_urlpath: str,
    output_urlpath: str,
    num_cores: int,
    use_gpu: bool = False,
    image: str = "mskmind/qupath-stardist:0.4.3",
    use_singularity: bool = False,
    max_heap_size: str = "64G",
    storage_options: dict = {},
    output_storage_options: dict = {},
) -> pd.DataFrame:
    """Run stardist using qupath CLI

    Args:
        slide_urlpath (str): url/path to slide image (virtual slide formats compatible with openslide, .svs, .tif, .scn, ...)
        output_urlpath (str): output url/path
        num_cores (int): Number of cores to use for CPU parallelization
        use_gpu (bool): use GPU
        use_singularity (bool): use singularity instead of docker
        max_heap_size (str): maximum heap size to pass to java options
        storage_options (dict): storage options to pass to reading functions

    Returns:
        pd.DataFrame: cell detections
    """
    fs, slide_path = fsspec.core.url_to_fs(slide_urlpath, **storage_options)

    ofs, output_path = fsspec.core.url_to_fs(output_urlpath, **output_storage_options)

    if ofs.protocol == 'file' and not ofs.exists(output_path):
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
    for line in executor:
        print(line)

    stardist_output = Path(output_path) / "cell_detections.tsv"

    df = pd.read_csv(stardist_output, sep="\t")
    df.index = "cell-" + df.index.astype(int).astype(str)
    df.index.rename("cell_id", inplace=True)

    df = df.rename(
        columns={"Centroid X µm": "x_coord", "Centroid Y µm": "y_coord"}
    )  # x,ys follow this convention

    return df


def fire_cli():
    fire.Fire(
        {
            "simple": stardist_simple,
            "cell-lymphocyte": stardist_cell_lymphocyte,
        }
    )


if __name__ == "__main__":
    fire_cli()
