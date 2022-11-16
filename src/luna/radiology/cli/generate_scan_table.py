from pathlib import Path
from medpy.io import save, load
import dicom2nifti
import os
import click
import pandas as pd
import dask.dataframe as dd
import numpy as np
from typing import Dict, List, Optional, Tuple

from luna.common.utils import cli_runner
from luna.common.custom_logger import init_logger
from luna.common.utils import generate_uuid

logger = init_logger()

_params_ = [('raw_data_path', str), ('mapping_csv_path', str), ('output_dir', str),
            ('scan_table_path', str), ('npartitions', int), ('subset', bool)]

def find_z_nbound(dim: Tuple) -> Tuple[int, int]:
    """
    Return z, number of series bound

    Args:
        dim (Tuple): shape of the 4D bound series.
    """
    # axial (x, z, nbound, y)
    if dim[0] == dim[3]:
        return dim[1], dim[2]
    # sagittal (x, y, nbound, z)
    if dim[0] == dim[1]:
        return dim[3], dim[2]


def subset_bound(src_path: str, output_path: str, index: int):
    """
    Subset bound series and return the output file path.

    Args:
        src_path (str): path to the bound series
        output_path (str): path to save the subset output
        index (int): index to pull out from the bound 4D scan
    """
    try:
        file_path = src_path.split(':')[-1]
        data, header = load(file_path)
        subset = data[:, :, index, :]
        # re-arrange the array
        subset = np.swapaxes(np.swapaxes(subset, 1, 2), 0, 1)
        save(subset, output_path, header)
        return output_path

    except Exception as err:
        logger.error(output_path, err)
        return None


def subset_series(dim: str, src_path: str) -> str:
    """
    Subset scans if the scan is a bound series (with 4 dimensions).

    Args:
        dim (str): shape of the scan
        src_path (str): path to the scan

    Returns:
        str: path to the subset output
    """
    dim = eval(dim)
    posix_path = Path(src_path)

    if len(dim) == 4:
        # axial (x, z, n_bound_series, y)
        # sag (x, y, n_bound_series, z)
        z, nbound = find_z_nbound(dim)

        # if 3 series are bound, assume there are three post contrasts bound
        if nbound == 3:
            index = 0
        # if more than 3 series are bound, assume the first one is pre
        elif nbound > 3:
            index = 1

        output_path = os.path.join(str(posix_path.parent), f"subset_{index}_" + str(posix_path.name))
        if not os.path.exists(output_path):
            subset_bound(src_path, output_path, index)
            logger.info("Saved ", output_path)

    return output_path


def dicom_to_nifti(row: pd.DataFrame, raw_data_path: str, output_dir: str, subset=False)\
        -> pd.DataFrame:
    """
    Convert dicoms to nii.gz format

    Args:
        row (pd.DataFrame): dataframe row with path, accession_number columns
        raw_data_path (str): path to dicom files
        output_dir (str): path to save scans
        subset (Optional, boolean): True to extract first post contrast in bound series

    Returns:
        pd.DataFrame: row populated with record_uuid, dim, path, subset_path
    """
    accession_number = row.accession_number
    series_number = row.series_number

    dicom_dir = f"{raw_data_path}/{accession_number}/SCANS/{series_number}/DICOM"
    scan_dir = f"{output_dir}/{accession_number}"

    logger.info(scan_dir)
    os.makedirs(scan_dir, exist_ok=True)

    logger.info(f"Processing : {dicom_dir}")
    try:
        scan_file = None
        for fn in os.listdir(scan_dir):
            if fn.endswith("nii.gz") and fn.startswith(str(series_number)):
                scan_file = fn
                break
        if not scan_file:
            dicom2nifti.convert_directory(dicom_dir, scan_dir, reorient=False)

        print(os.listdir(scan_dir))
        for fn in os.listdir(scan_dir):
            if fn.endswith("nii.gz") and fn.startswith(str(series_number)):
                scan_file = fn
                break

        scan_file_path = os.path.join(scan_dir, scan_file)
        data, header = load(scan_file_path)

        row["record_uuid"] = generate_uuid(scan_file_path, ["SCAN"])
        row["path"] = scan_file_path
        row["dim"] = str(data.shape)

        if len(data.shape) == 4 and subset:
            row["subset_path"] = subset_series(row.dim, row.path)

    except Exception as err:
        logger.error(f"Failed {accession_number}: {err}")

    return row


@click.command()
@click.option('-r', '--raw_data_path', help="path to raw data. e.g /data/radiology/project_name/dicoms",
              type=click.Path())
@click.option('-c', '--mapping_csv_path', help="csv with AccessionNumber, SeriesNumber columns",
              type=click.Path())
@click.option('-o', '--output_dir', help="path to scan output folder")
@click.option('-t', '--scan_table_path', help="path to save scan table")
@click.option('-s', '--subset', help="extract first post contrast in bound series", is_flag=True)
@click.option('-n', '--npartitions', help="npartitions for parallelization", default=20, show_default=True)
@click.option('-m', '--method_param_path', help='path to a metadata json/yaml file with method parameters to reproduce results',
              type=click.Path())
def cli(**cli_kwargs):
    """
    Convert dicoms to nii.gz format and optionally subset bound series.

    dicom2nifti is used to convert bound series into 4D volumes.
    For more simple dicom to scan function, see luna.radiology.cli.dicom_to_itk.
    """
    cli_runner(cli_kwargs, _params_, generate_scan_table)

def generate_scan_table(raw_data_path: str, mapping_csv_path: str, output_dir: str,
                        scan_table_path: str, subset=False, npartitions=20):
    """
    Convert dicoms to nii.gz format and optionally subset bound series.
    Save scan table and method params.

    Args:
        dicom_parquet_path (str): path to dicom parquet table
        mapping_csv_path (str): csv with AccessionNumber, SeriesNumber columns
        output_dir (str): path to scan output folder
        scan_table_path (str): path to save scan table
        subset (Optional, boolean): True to extract first post contrast in bound series
        npartitions (Optional, int): number of partitions for parallelization. Default is 20
    """
    input_params = locals()

    # load accession/series mapping csv
    scan_map = pd.read_csv(mapping_csv_path)
    scan_map = scan_map[['AccessionNumber', 'SeriesNumber']]
    df = scan_map \
        .rename({'AccessionNumber': 'accession_number', 'SeriesNumber': 'series_number'}, axis=1) \
        .astype(str)

    # convert to nii.gz
    df["create_ts"] = pd.Timestamp.now()
    df["dim"] = ""
    df["path"] = ""
    df["record_uuid"] = ""
    df["subset_path"] = ""

    ddf = dd.from_pandas(df, npartitions=npartitions)
    df = ddf.apply(lambda x: dicom_to_nifti(x, raw_data_path, output_dir, subset=subset), axis=1,
                   meta=ddf).compute()
    logger.info(df)

    # save table as parquet
    df = df.replace("", np.nan)
    df = df.dropna(subset=["record_uuid"])
    df.to_parquet(scan_table_path)

    return {
        'table_path': scan_table_path,
        'n_records': len(df)
    }

if __name__ == "__main__":
    cli()
