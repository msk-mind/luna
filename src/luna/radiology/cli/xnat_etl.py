# ===========================================
# Usual Imports
import logging
import os

import click
import pandas as pd
import pyxnat

# ===========================================
# Custom Imports
import xmltodict
from dask.delayed import delayed
from dask.distributed import Client

from luna.common.custom_logger import init_logger
from luna.common.utils import cli_runner

init_logger()
logger = logging.getLogger("xnat_etl")

_params_ = [
    ("input_xnat_url", str),
    ("username", str),
    ("password", str),
    ("num_cores", int),
    ("project_name", str),
    ("debug_limit", int),
    ("output_dir", str),
]


@click.command()
@click.argument("input_xnat_url", nargs=1)
@click.option(
    "-u",
    "--username",
    required=False,
    help="DSA username, can be inferred from XNAT_USERNAME",
)
@click.option(
    "-p",
    "--password",
    required=False,
    help="DSA password, should be inferred from XNAT_PASSWORD",
)
@click.option(
    "-p",
    "--project_name",
    required=False,
    help="project name to which slides are assigned or associated",
)
@click.option(
    "-dl",
    "--debug_limit",
    required=False,
    default=-1,
    help="limit number of slides process, for debugging, no_write is automatically enabled",
)
@click.option(
    "-nc", "--num_cores", required=False, help="Number of cores to use", default=4
)
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
    help="Optional dataset identifier to add results to",
)
def cli(**cli_kwargs):
    """Ingest images from XNAT

    \b
    Inputs:
        xnat_url: XNAT URL

    \b
    Outputs:
        scan_table: a dataset representing the scan ingestion job

    \b
    Example:
        xnat_etl https://xnat.my.insitution.org
            --project_name 12-345
            --num_cores 8
            -o /data/RAD-12-345/
            -dsid XNAT_TABLE

    python3 -m luna.radiology.cli.xnat_etl https://xnat.mskcc.org -p 16-1335 -s _ -nc 16 --no-write -o . -dl 8 -dsid XNAT_TABLE
    """
    cli_runner(cli_kwargs, _params_, xnat_etl)


def xnat_etl(
    input_xnat_url,
    username,
    password,
    project_name,
    num_cores,
    debug_limit,
    output_dir,
):
    """ """
    xnat = pyxnat.Interface(input_xnat_url, user=username, password=password)

    p = xnat.select.project(project_name)

    logger.info(f"{p}")

    with Client(
        host=os.environ["HOSTNAME"],
        n_workers=((num_cores + 7) // 8),
        threads_per_worker=8,
    ) as client:
        logger.info(f"Client: {client} ")
        logger.info(f"Dashboard: {client.dashboard_link}")

        dfs = [
            delayed(get_patient_scans)(s, output_dir)
            for i, s in enumerate(p.subjects())
            if (i < debug_limit or debug_limit < 0)
        ]
        df_project = delayed(pd.concat)(dfs).compute()

        logger.info(df_project)

    output_file = os.path.join(output_dir, f"xnat_data_{project_name}.parquet")
    df_project.to_parquet(output_file)

    return {
        "feature_data": output_file,
        "segment_keys": {
            "xnat_project_id": project_name,
        },
    }


def experiment_scans_to_df(s, experiment_id, output_dir):
    logger.info(f"Processing experiment id={experiment_id}")

    e = s.experiment(experiment_id)

    d_scans_list = [xmltodict.parse(scan.get()) for scan in e.scans()]
    d_scans_types = [list(d_scan.items())[0][0] for d_scan in d_scans_list]
    d_scans_list = [list(d_scan.items())[0][1] for d_scan in d_scans_list]

    df_scans = pd.json_normalize(d_scans_list, sep="_")
    df_scans["xnatScanType"] = d_scans_types
    df_scans["experiment_id"] = experiment_id

    for scan_id in df_scans["@ID"]:

        df_scan = df_scans.loc[df_scans["@ID"] == scan_id]

        dest_dir = os.path.join(output_dir, experiment_id, scan_id)
        os.makedirs(dest_dir, exist_ok=True)
        r = e.scan(scan_id).resource("DICOM")
        logger.info(
            f"Scan: {experiment_id} {scan_id} {df_scan['@type'].item()}, Resource: {r}"
        )

        if not len(list(r.files())):
            continue  # Will be >0 if there are files to download

        r.get(dest_dir, extract=True)

        df_scans.loc[df_scans["@ID"] == scan_id, "dicom_folder"] = dest_dir + "/DICOM"

    return df_scans


def get_patient_scans(s, output_dir):
    logger.info(f"Processing subject {s}")

    d_exps_list = [xmltodict.parse(experiment.get()) for experiment in s.experiments()]
    d_exps_types = [list(d_exp.items())[0][0] for d_exp in d_exps_list]
    d_exps_list = [list(d_exp.items())[0][1] for d_exp in d_exps_list]

    df_exps = pd.json_normalize(d_exps_list, sep="_").drop(
        columns=["xnat:scans_xnat:scan"]
    )
    df_exps["xnatExpType"] = d_exps_types

    df_scans = [
        experiment_scans_to_df(s, exp_id, output_dir) for exp_id in df_exps["@ID"]
    ]

    df_scans = pd.concat(df_scans)

    df_subject = df_scans.set_index("experiment_id").join(
        df_exps.set_index("@ID").drop(columns=df_scans.columns, errors="ignore")
    )

    # Do some normalization
    df_subject = df_subject.rename(
        columns={
            "xnat:dcmAccessionNumber": "radiology_accession_number",
            "xnat:dcmPatientName": "radiology_patient_id",
            "@ID": "radiology_series_number",
            "@type": "scan_type_description",
            "@UID": "radiology_series_instance_uuid",
            "@project": "xnat_project_id",
        }
    )
    return df_subject


if __name__ == "__main__":
    cli(auto_envvar_prefix="XNAT")
