import fire
import glob
from typing import List, Union
import pandas as pd
from pathlib import Path
import fsspec
from fsspec import open

from luna.common.utils import get_config, timed, save_metadata
from luna.common.models import ShapeFeaturesSchema


@timed
@save_metadata
def cli(
    shape_features_urlpaths: Union[str,List[str]] = "???",
    output_urlpath: str = ".",
    storage_options: dict = {},
    output_storage_options: dict = {},
    local_config: str = "",
):
    """Merges shape features dataframes

    Args:
        shape_features_urlpaths (List[str]): URL/paths to shape featurs parquet files
        output_urlpath (str): URL/path to output parquet file
        storage_options (dict): storage options to pass to reading functions
        output_storage_options (dict): storage options to pass to writing functions
        local_config (str): local config yaml file

    Returns:
        dict: output paths and the number of features generated
    """
    config = get_config(vars())

    dfs = []  # type: list[str]
    if type(config["shape_features_urlpaths"]) == list:
        for urlpath in config['shape_features_urlpaths']:
            fs, path = fsspec.core.url_to_fs(urlpath, **config['storage_options'])
            with fs.open(path, 'rb') as of:
                df = pd.read_parquet(of)
            ShapeFeaturesSchema.validate(df)
            dfs.append(df)
    else:
        fs, path_prefix = fsspec.core.url_to_fs(config['shape_features_urlpaths'],
                                                **config['storage_options'])
        for path in fs.glob(f"{path_prefix}/**/shape_features.parquet"):
            with fs.open(path, 'rb') as of:
                df = pd.read_parquet(of)
            ShapeFeaturesSchema.validate(df)
            dfs.append(df)

    df = pd.concat(dfs)
    fs, path_prefix = fsspec.core.url_to_fs(config['output_urlpath'],
                                            **config['output_storage_options'])
    path = Path(path_prefix) / 'long_shape_features.parquet'

    with fs.open(path, 'wb', **config["output_storage_options"]) as of:
        df.to_parquet(of)

    df.variable = df.variable.str.replace('µ','u').replace(r'(: |:)', ' ', regex=True).replace('[^a-zA-Z0-9 \n]', '', regex=True)
    wide_path = Path(path_prefix) / 'wide_shape_features.parquet'
    wide_df = df.pivot(index="slide_id", columns=["Parent", "Class", "variable"], values="value")
    with fs.open(wide_path, 'wb', **config["output_storage_options"]) as of:
        wide_df.to_parquet(of)

    return  {
        "long_shape_features": fs.unstrip_protocol(str(path)),
        "wide_shape_features": fs.unstrip_protocol(str(wide_path)),
        "num_features": len(wide_df.columns),
    }


def fire_cli():
    fire.Fire(cli)

if __name__ == "__main__":
    fire_cli()


