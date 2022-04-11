# General imports
import os
import logging
import click
import pandas as pd
from pathlib import Path

import luna.common.stats
from luna.common.custom_logger import init_logger
from luna.common.utils import cli_runner

init_logger()
logger = logging.getLogger("extract_tile_statistics")  # Add CLI tool name

_params_ = [("input_slide_tiles", str), ("output_dir", str)]


@click.command()
@click.argument("input_slide_tiles", nargs=1)
@click.option(
    "-o",
    "--output_dir",
    required=False,
    help="path to output directory to save results",
)
@click.option(
    "-m",
    "--method_param_path",
    required=False,
    help="path to a metadata json/yaml file with method parameters to reproduce results",
)
@click.option(
    "-dsid",
    "--dataset_id",
    required=False,
    help="Optional dataset identifier to add tabular output to",
)
def cli(**cli_kwargs):
    """Extracts statistics over tiles

    \b
    Inputs:
        input: input data
    \b
    Outputs:
        output data
    \b
    Example:
        CLI_TOOL ./slides/10001.svs ./halo/10001.job18484.annotations
            -an Tumor
            -o ./masks/10001/
    """
    cli_runner(cli_kwargs, _params_, extract_tile_statistics)


def extract_tile_statistics(input_slide_tiles, output_dir):
    """Extracts statistics over tiles

    Args:
        input_slide_tiles (str): path to input data
        output_dir (str): output/working directory

    Returns:
        dict: metadata about function call
    """

    df = (
        pd.read_parquet(input_slide_tiles)
        .reset_index()
        .set_index("address")
        .drop(columns=["x_coord", "y_coord", "tile_size", 'xy_extent', 'tile_units'], errors='ignore')
    )
    print(df.columns)

    dict_feature_data = {}

    for col in df.columns:
        dict_feature_data.update(
            luna.common.stats.compute_stats_1d(pd.to_numeric(df[col]), col)
        )

    df_feature_data = pd.DataFrame([dict_feature_data])

    output_feature_file = os.path.join(
        output_dir, Path(input_slide_tiles).stem + "_tile_stats.parquet"
    )

    logger.info(df_feature_data)

    df_feature_data.to_parquet(output_feature_file)

    properties = {"feature_data": output_feature_file}

    return properties


if __name__ == "__main__":
    cli()
