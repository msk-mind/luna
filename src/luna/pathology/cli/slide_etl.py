# General imports
import uuid
from pathlib import Path

import fire
import fsspec
import pandas as pd
from dask.distributed import Client, get_client, progress
from fsspec import open  # type: ignore
from loguru import logger
from multimethod import multimethod
from pandera.typing import DataFrame
from tiffslide import TiffSlide

from luna.common.models import Slide, SlideSchema
from luna.common.utils import apply_csv_filter, get_config, timed
from luna.pathology.common.utils import (
    get_downscaled_thumbnail,
    get_scale_factor_at_magnification,
    get_stain_vectors_macenko,
)

VALID_SLIDE_EXTENSIONS = [".svs", ".scn", ".tif"]


@timed
def cli(
    slide_urlpath: str = "???",
    project_name: str = "",
    comment: str = "",
    subset_csv_urlpath: str = "",
    debug_limit: int = 0,
    output_urlpath: str = "",
    storage_options: dict = {},
    output_storage_options: dict = {},
    local_config: str = "",
):
    """Ingest slide by adding them to a file or s3 based storage location and generating metadata about them


    Args:
        slide_url (str): path to slide image
        project_name (str): project name underwhich the slides should reside
        comment (str): comment and description of dataset
        subset_csv_urlpath (str): url/path to subset csv
        storage_options (dict): storage options to pass to reading functions
        output_urlpath (str): url/path to output table
        output_storage_options (dict): storage options to pass to writing functions
        local_config (str): url/path to YAML config file


    Returns:
        slide (Slide): slide object
    """

    config = get_config(vars())
    filesystem, slide_path = fsspec.core.url_to_fs(
        config["slide_urlpath"], **config["storage_options"]
    )
    slide_paths = []  # type: list[str]
    if any([slide_path.endswith(ext) for ext in VALID_SLIDE_EXTENSIONS]):
        slide_paths += slide_path
    else:
        for ext in VALID_SLIDE_EXTENSIONS:
            slide_paths += filesystem.glob(f"{slide_path}/*{ext}")

    if config["subset_csv_urlpath"]:
        slide_paths = apply_csv_filter(
            slide_paths, config["subset_csv_urlpath"], config["storage_options"]
        )
    if config["debug_limit"] > 0:
        slide_paths = slide_paths[: config["debug_limit"]]

    if len(slide_paths) == 0:
        return None

    slide_urls = [filesystem.unstrip_protocol(slide_path) for slide_path in slide_paths]

    df = slide_etl(
        slide_urls,
        config["project_name"],
        config["comment"],
        config["storage_options"],
        config["output_urlpath"],
        config["output_storage_options"],
    )

    # rebase_schema_numeric(df)

    logger.info(df)
    if config["output_urlpath"]:
        output_filesystem, output_path = fsspec.core.url_to_fs(
            config["output_urlpath"], **config["output_storage_options"]
        )
        f = Path(output_path) / f"slide_ingest_{config['project_name']}.parquet"
        with output_filesystem.open(f, "wb") as of:
            df.to_parquet(of)


@multimethod
def slide_etl(
    slide_urls: list[str],
    project_name: str,
    comment: str = "",
    storage_options: dict = {},
    output_urlpath: str = "",
    output_storage_options: dict = {},
) -> DataFrame:
    client = get_client()
    sb = SlideBuilder(storage_options, output_storage_options=output_storage_options)

    futures = [
        client.submit(
            sb.get_slide,
            url,
            project_name=project_name,
            comment=comment,
        )
        for url in slide_urls
    ]
    progress(futures)
    slides = client.gather(futures)
    if output_urlpath:
        futures = [
            client.submit(
                sb.copy_slide,
                slide,
                output_urlpath,
            )
            for slide in slides
        ]
    return DataFrame[SlideSchema](
        pd.json_normalize([x.__dict__ for x in client.gather(futures)])
    )


@multimethod
def slide_etl(
    slide_url: str,
    project_name: str,
    comment: str = "",
    storage_options: dict = {},
    output_urlpath: str = "",
    output_storage_options: dict = {},
) -> Slide:
    """Ingest slide by adding them to a file or s3 based storage location and generating metadata about them

    Args:
        slide_url (str): path to slide image
        project_name (str): project name underwhich the slides should reside
        comment (str): comment and description of dataset
        storage_options (dict): storage options to pass to reading functions
        output_urlpath (str): url/path to output table
        output_storage_options (dict): storage options to pass to writing functions


    Returns:
        slide (Slide): slide object
    """

    sb = SlideBuilder(storage_options, output_storage_options=output_storage_options)

    slide = sb.get_slide(slide_url, project_name=project_name, comment=comment)
    if output_urlpath:
        slide = sb.copy_slide(slide, output_urlpath)
    return slide


class SlideBuilder:
    def __init__(self, storage_options: dict = {}, output_storage_options: dict = {}):
        self.storage_options = storage_options
        self.output_storage_options = output_storage_options

    def __generate_properties(self, slide, url):
        with open(url, **self.storage_options) as f:
            s = TiffSlide(f)
            slide.properties = s.properties
            try:
                to_mag_scale_factor = get_scale_factor_at_magnification(
                    s, requested_magnification=1
                )
                sample_arr = get_downscaled_thumbnail(s, to_mag_scale_factor)
                stain_vectors = get_stain_vectors_macenko(sample_arr)
                slide.channel0_R = stain_vectors[0, 0]
                slide.channel0_G = stain_vectors[0, 1]
                slide.channel0_B = stain_vectors[0, 2]
                slide.channel1_R = stain_vectors[1, 0]
                slide.channel1_G = stain_vectors[1, 1]
                slide.channel1_B = stain_vectors[1, 2]
            except Exception as err:
                logger.warning(f"Couldn't get stain vectors: {url} - {err}")

    def copy_slide(self, slide, output_urlpath, chunksize=50000000):
        new_slide = slide.copy()
        name = Path(slide.url).name
        fs, output_path = fsspec.core.url_to_fs(
            output_urlpath, **self.output_storage_options
        )
        p = Path(output_path) / name
        with open(slide.url, "rb", **self.storage_options) as f1:
            with fs.open(p, "wb") as f2:
                while True:
                    data = f1.read(chunksize)
                    if not data:
                        break
                    f2.write(data)
        new_slide.url = fs.unstrip_protocol(str(p))
        return new_slide

    def get_slide(self, url, project_name="", comment="") -> Slide:
        """Extract openslide properties and write slide to storage location

        Args:
            path (string): path to slide image

        Returns:
            slide (Slide): slide object
        """

        fs, path = fsspec.core.url_to_fs(url, **self.storage_options)

        id = Path(path).stem
        size = fs.du(path)
        slide = Slide(
            id=id,
            project_name=project_name,
            comment=comment,
            slide_size=size,
            url=url,
            uuid=str(uuid.uuid3(uuid.NAMESPACE_URL, url)),
        )

        self.__generate_properties(slide, url)

        return slide

    # Eventually it might be nice to automatically detect the stain type (at least H&E vs. DAB vs. Other)
    # def estimate_stain(self, slide):


if __name__ == "__main__":
    client = Client()
    fire.Fire(cli)
