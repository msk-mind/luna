# General imports
import os
from pathlib import Path

import fire
import signal
import fsspec
import pandas as pd
from loguru import logger

import docker
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
        storage_options (dict): storage options to pass to reading functions
        output_storage_options (dict): storage options to pass to writing functions

    Returns:
        pd.DataFrame: metadata about function call
    """
    config = get_config(vars())

    df = stardist_simple_main(
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

    logger.info("generated cell data:")
    logger.info(df)

    properties = {
        "cell_objects": output_header_file,
        "feature_data": output_header_file,
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
        storage_options (dict): storage options to pass to reading functions
        output_storage_options (dict): storage options to pass to writing functions

    Returns:
        pd.DataFrame: cell detections
    """
    fs, slide_path = fsspec.core.url_to_fs(slide_urlpath, **storage_options)

    ofs, output_path = fsspec.core.url_to_fs(output_urlpath, **output_storage_options)

    if ofs.protocol == 'file' and not os.path.exists(output_path):
        os.makedirs(output_path)

    slide_filename = Path(slide_path).name
    docker_image = "mskmind/qupath-stardist:0.4.3"
    command = f"QuPath script --image /inputs/{slide_filename} --args [cellSize={cell_expansion_size},imageType={image_type},{debug_opts}] /scripts/stardist_simple.groovy"
    logger.info("Launching docker container:")
    logger.info(
        f"\tvolumes={slide_urlpath}:'/inputs/{slide_filename}', {slide_path}:'/output_dir'"
    )
    logger.info(f"\tnano_cpus={int(num_cores * 1e9)}")
    logger.info(f"\timage='{docker_image}'")
    logger.info(f"\tcommand={command}")

    docker_kwargs = dict(
        user=os.getuid(),
        volumes={
            slide_path: {"bind": f"/inputs/{slide_filename}", "mode": "ro"},
            output_path: {"bind": "/output_dir", "mode": "rw"},
        },
        nano_cpus=int(num_cores * 1e9),
        image=docker_image,
        command=command,
        detach=True,
    )
    if use_gpu:
        docker_kwargs['device_requests'] = [docker.types.DeviceRequest(device_ids=["0"], capabilities=[['gpu']])]

    client = docker.from_env()
    container = client.containers.run(**docker_kwargs)
        
    def handler_stop_signals(signum, frame):
        nonlocal container
        logger.info("Received kill signal, stopping container...")
        container.stop()

    signal.signal(signal.SIGTERM, handler_stop_signals)
    signal.signal(signal.SIGINT, handler_stop_signals)

    for line in container.logs(stream=True):
        print(line.decode(), end="")

    result = container.wait()
    container.remove()

    if result['StatusCode'] != 0:
        logger.error(f"Docker container returned non-zero: {result['StatusCode']}")
        raise docker.errors.DockerException(f"command: {command}\nimage: {docker_image}\nreturn code: {result['StatusCode']}")

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
    storage_options: dict = {},
    output_storage_options: dict = {},
):
    """Run stardist using qupath CLI

    Args:
        input_slide_image (str): path to slide image (virtual slide formats compatible with openslide, .svs, .tif, .scn, ...)
        num_cores (int): Number of cores to use for CPU parallelization
        output_urlpath (str): output url/path
        storage_options (dict): storage options to pass to reading functions
        output_storage_options (dict): storage options to pass to writing functions

    Returns:
        pd.DataFrame: cell detections
    """
    config = get_config(vars())

    df = stardist_cell_lymphocyte_main(
        config["slide_urlpath"],
        config["output_urlpath"],
        config["num_cores"],
        config["use_gpu"],
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

    logger.info("generated cell data:")
    logger.info(df)

    properties = {
        "cell_objects": output_header_file,
        "feature_data": output_header_file,
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
    storage_options: dict = {},
    output_storage_options: dict = {},
) -> pd.DataFrame:
    """Run stardist using qupath CLI

    Args:
        input_slide_image (str): path to slide image (virtual slide formats compatible with openslide, .svs, .tif, .scn, ...)
        num_cores (int): Number of cores to use for CPU parallelization
        output_urlpath (str): output url/path
        storage_options (dict): storage options to pass to reading functions

    Returns:
        pd.DataFrame: cell detections
    """
    fs, slide_path = fsspec.core.url_to_fs(slide_urlpath, **storage_options)

    ofs, output_path = fsspec.core.url_to_fs(output_urlpath, **output_storage_options)

    if ofs.protocol == 'file' and not os.path.exists(output_path):
        os.makedirs(output_path)

    qupath_cmd = "QuPath-cpu"
    if use_gpu:
        qupath_cmd = "QuPath-gpu"

    slide_filename = Path(slide_path).name
    docker_image = "mskmind/qupath-stardist:0.4.3"
    command = f"{qupath_cmd} script  --image /inputs/{slide_filename} /scripts/stardist_nuclei_and_lymphocytes.groovy"
    logger.info("Launching docker container:")
    logger.info(
        f"\tvolumes={slide_path}:'/inputs/{slide_filename}', {output_path}:'/output_dir'"
    )
    logger.info(f"\tnano_cpus={int(num_cores * 1e9)}")
    logger.info(f"\timage='{docker_image}'")
    logger.info(f"\tcommand={command}")

    docker_kwargs = dict(
        user=os.getuid(),
        volumes={
            slide_path: {"bind": f"/inputs/{slide_filename}", "mode": "ro"},
            output_path: {"bind": "/output_dir", "mode": "rw"},
        },
        nano_cpus=int(num_cores * 1e9),
        image=docker_image,
        command=command,
        detach=True,
    )
    if use_gpu:
        docker_kwargs['device_requests'] = [docker.types.DeviceRequest(device_ids=["0"], capabilities=[['gpu']])]
        

    client = docker.from_env()
    container = client.containers.run(**docker_kwargs)
    def handler_stop_signals(signum, frame):
        nonlocal container
        logger.info("Received kill signal, stopping container...")
        container.stop()

    signal.signal(signal.SIGTERM, handler_stop_signals)
    signal.signal(signal.SIGINT, handler_stop_signals)

    for line in container.logs(stream=True):
        print(line.decode(), end="")

    result = container.wait()
    container.remove()

    if result['StatusCode'] != 0:
        logger.error(f"Docker container returned non-zero: {result['StatusCode']}")
        raise docker.errors.DockerException(f"command: {command}\nimage: {docker_image}\nreturn code: {result['StatusCode']}")

    stardist_output = Path(output_path) / "cell_detections.tsv"

    df = pd.read_csv(stardist_output, sep="\t")
    df.index = "cell-" + df.index.astype(int).astype(str)
    df.index.rename("cell_id", inplace=True)

    df = df.rename(
        columns={"Centroid X µm": "x_coord", "Centroid Y µm": "y_coord"}
    )  # x,ys follow this convention

    return df


if __name__ == "__main__":

    fire.Fire(
        {
            "simple": stardist_simple,
            "cell-lymphocyte": stardist_cell_lymphocyte,
        }
    )
