# General imports
import logging
import os
import tempfile
from pathlib import Path

import fire
import fsspec
import pandas as pd

import docker
from luna.common.custom_logger import init_logger
from luna.common.utils import get_config

init_logger()
logger = logging.getLogger("run_stardist_cell_detection")


def cli(
    slide_urlpath: str = "???",
    cell_expansion_size: float = "???",  # type: ignore
    image_type: str = "???",
    output_urlpath: str = ".",
    debug_opts: str = "",
    num_cores: int = 1,
    storage_options: dict = {},
    output_storage_options: dict = {},
    local_config: str = "",
):
    """Run stardist using qupath CLI within a docker container

    Note: containers are spawned with a root process, and will not exit if the CLI tool is aborted.

    Note: Stardist cell coordinates are in microns. To convert to pixel coordinates, divide by microns-per-pixel (mpp)

    TODO: Improve handling of containers, move to singularity, and/or ensure graceful exits

    \b
    Inputs:
        input_slide_image: slide image (virtual slide formats compatible with openslide, .svs, .tif, .scn, ...)
    \b
    Outputs:
        cell_objects
    \b
    Example:
        run_stardist_cell_detection /absolute/path/to/10001.svs
            -cs 8 -it BRIGHTFIELD_H_DAB -nc 8
            -o /absolute/path/to/working_dir
    """
    config = get_config(vars())

    df = run_stardist_cell_detection(
        config["slide_urlpath"],
        config["cell_expansion_size"],
        config["image_type"],
        config["output_urlpath"],
        config["debug_opts"],
        config["num_cores"],
        config["storage_options"],
        config["output_storage_options"],
    )

    slide_id = Path(config["slide_urlpath"]).stem

    fs, output_path = fsspec.core.url_to_fs(
        config["output_urlpath"], **config["output_storage_options"]
    )
    output_header_file = Path(output_path) / f"{slide_id}_cell_objects.parquet"
    with fs.open(output_header_file, "wb") as of:
        df.to_parquet(of)

    logger.info("Generated cell data:")
    logger.info(df)

    properties = {
        "cell_objects": output_header_file,
        "feature_data": output_header_file,
        "spatial": True,
        "total_cells": len(df),
        "segment_keys": {"slide_id": slide_id},
    }

    return properties


def run_stardist_cell_detection(
    slide_urlpath: str,
    cell_expansion_size: float,
    image_type: str,
    output_urlpath: str,
    debug_opts: str,
    num_cores: int,
    storage_options: dict,
    output_storage_options: dict,
):
    """Run stardist using qupath CLI

    Args:
        input_slide_image (str): path to slide image (virtual slide formats compatible with openslide, .svs, .tif, .scn, ...)
        cell_expansion_size (float): size in pixels to expand cell cytoplasm
        num_cores (int): Number of cores to use for CPU parallelization
        image_type (str): qupath image type (BRIGHTFIELD_H_DAB)
        output_urlpath (str): output url/path
        debug_opts (str): debug options passed as arguments to groovy script
        storage_options (dict): storage options to pass to reading functions

    Returns:
        dict: metadata about function call
    """
    fs, slide_path = fsspec.core.url_to_fs(slide_urlpath, **storage_options)
    if fs.protocol == "file":
        local_file = slide_path
    else:
        fs = fsspec.filesystem("simplecache", target_protocol=fs.protocol)
        of = fs.open(slide_path, "r")
        local_file = of.name

    ofs, output_path = fsspec.core.url_to_fs(output_urlpath, **output_storage_options)
    if ofs.protocol == "file":
        local_working_path = output_path
    else:
        tmp_dir = tempfile.TemporaryDirectory()
        local_working_path = tmp_dir.name

    slide_filename = Path(slide_path).name
    docker_image = "mskmind/qupath-stardist"
    command = f"QuPath script --image /inputs/{slide_filename} --args [cellSize={cell_expansion_size},imageType={image_type},{debug_opts}] /scripts/stardist_simple.groovy"
    logger.info("Launching docker container:")
    logger.info(
        f"\tvolumes={slide_urlpath}:'/inputs/{slide_filename}', {local_working_path}:'/output_dir'"
    )
    logger.info(f"\tnano_cpus={int(num_cores * 1e9)}")
    logger.info(f"\timage='{docker_image}'")
    logger.info(f"\tcommand={command}")

    os.makedirs(local_working_path, exist_ok=True)

    client = docker.from_env()
    container = client.containers.run(
        volumes={
            local_file: {"bind": f"/inputs/{slide_filename}", "mode": "ro"},
            local_working_path: {"bind": "/output_dir", "mode": "rw"},
        },
        nano_cpus=int(num_cores * 1e9),
        image=docker_image,
        command=command,
        detach=True,
    )
    if ofs.protocol != "file":
        ofs.copy(local_working_path, output_path)

    for line in container.logs(stream=True):
        print(line.decode(), end="")

    stardist_output = Path(local_working_path) / "cell_detections.tsv"

    df = pd.read_csv(stardist_output, sep="\t")
    df.index = "cell-" + df.index.astype(int).astype(str)
    df.index.rename("cell_id", inplace=True)

    df = df.rename(
        columns={"Centroid X µm": "x_coord", "Centroid Y µm": "y_coord"}
    )  # x,ys follow this convention

    return df


if __name__ == "__main__":
    fire.Fire(cli)
