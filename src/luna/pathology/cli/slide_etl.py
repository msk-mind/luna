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
    slide_dir: str = "???",
    project_name: str = "",
    comment: str = "",
    subset_csv: str = "",
    debug_limit: int = 0,
    output_url: str = "",
    storage_options: dict = {},
    output_storage_options: dict = {},
    local_config: str = "",
):
    """Ingest slides by adding them to a file or s3 based storage location and generating metadata about them

    Output schema follows [ slide_id (str), slide_uuid (str), readable (bool), comment, project, ingest_time (UTC), generation_date (UTC), <store data | data_url >, <openslide data>, <stain data> ]

    \b
    Inputs:
        input_slide_folder: a slide image folder to walk over to find slide images

    \b
    Outputs:
        slide_table: a dataset representing the slide ingestion job

    \b
    Example:
        slide_etl /source/pathology/images
            --project_name PATH-123-45
            --comment "H&E needle core biospies"
            --store_url s3://localhost:9000/
            --num_cores 8
            -o /data/PATH-123-45/table
    """
    config = get_config(vars())
    filesystem, urlpath = fsspec.core.url_to_fs(
        config["slide_dir"], **config["storage_options"]
    )
    slide_urlpaths = []  # type: list[str]
    for ext in VALID_SLIDE_EXTENSIONS:
        slide_urlpaths += filesystem.glob(f"{urlpath}/*{ext}")
    if config["subset_csv"]:
        slide_urlpaths = apply_csv_filter(slide_urlpaths, config["subset_csv"])
    if config["debug_limit"] > 0:
        slide_urlpaths = slide_urlpaths[: config["debug_limit"]]

    if len(slide_urlpaths) == 0:
        return None

    slide_urls = [
        filesystem.unstrip_protocol(slide_urlpath) for slide_urlpath in slide_urlpaths
    ]

    df = slide_etl(
        slide_urls,
        config["project_name"],
        config["comment"],
        config["storage_options"],
        config["output_url"],
        config["output_storage_options"],
    )

    # rebase_schema_numeric(df)

    logger.info(df)

    if config["output_url"]:
        output_filesystem, output_urlpath_prefix = fsspec.core.url_to_fs(
            config["output_url"], **config["output_storage_options"]
        )
        with output_filesystem.open(
            Path(output_urlpath_prefix)
            / f"slide_ingest_{config['project_name']}.parquet"
        ) as of:
            df.to_parquet(of)

    return df


@multimethod
def slide_etl(
    slide_urls: list[str],
    project_name: str,
    comment: str = "",
    storage_options: dict = {},
    output_url_prefix: str = "",
    output_storage_options: dict = {},
) -> DataFrame:
    client = get_client()
    sb = SlideBuilder(storage_options, output_storage_options=output_storage_options)

    futures = [
        client.submit(
            sb.get_slide,
            url,
            output_url_prefix,
            project_name=project_name,
            comment=comment,
        )
        for url in slide_urls
    ]
    progress(futures)
    return DataFrame[SlideSchema](
        pd.json_normalize([x.__dict__ for x in client.gather(futures)])
    )


@multimethod
def slide_etl(
    slide_url: str,
    project_name: str,
    comment: str = "",
    storage_options: dict = {},
    output_url_prefix: str = "",
    output_storage_options: dict = {},
) -> Slide:
    """Ingest slide by adding them to a file or s3 based storage location and generating metadata about them


    Args:
        slide_urlpath (str): path to slide image
        project_name (str): project name underwhich the slides should reside
        comment (str): comment and description of dataset
        debug_limit (int): cap number fo slides to process below this number (for debugging, testing, no_write is automatically enabled)
        output_dir (str): path to output table


    Returns:
        slide (Slide): slide object
    """

    sb = SlideBuilder(storage_options, output_storage_options=output_storage_options)

    slide = sb.get_slide(
        slide_url, output_url_prefix, project_name=project_name, comment=comment
    )
    return slide


class SlideBuilder:
    def __init__(self, storage_options: dict = {}, output_storage_options: dict = {}):
        self.storage_options = storage_options
        self.output_storage_options = output_storage_options

    def __generate_properties(self, slide, url):
        with open(url, **self.storage_options) as f:
            slide.properties = TiffSlide(f).properties
        return slide

    def __estimate_stain_type(self, slide, url):
        try:
            with open(url, **self.storage_options) as f:
                s = TiffSlide(f)
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
            return {}

    def get_slide(
        self, url, output_url_prefix, project_name="", comment="", chunksize=50000000
    ) -> Slide:
        """Extract openslide properties and write slide to storage location

        Args:
            path (string): path to slide image

        Returns:
            slide (Slide): slide object
        """

        fs, urlpath = fsspec.core.url_to_fs(url, **self.storage_options)

        id = Path(urlpath).stem
        size = fs.du(urlpath)
        slide = Slide(
            id=id,
            project_name=project_name,
            comment=comment,
            size=size,
            url=url,
            uuid=uuid.uuid3(uuid.NAMESPACE_URL, url),
        )

        self.__estimate_stain_type(slide, url)
        self.__generate_properties(slide, url)

        if output_url_prefix:
            output_url = output_url_prefix + id
            with fs.open(urlpath) as f1:
                with open(output_url, **self.output_storage_options) as f2:
                    while True:
                        data = f1.read(chunksize)
                        if not data:
                            break
                        f2.write(data)
            slide.url = output_url

        return slide

    # Eventually it might be nice to automatically detect the stain type (at least H&E vs. DAB vs. Other)
    # def estimate_stain(self, slide):


if __name__ == "__main__":
    client = Client()
    fire.Fire(cli)
