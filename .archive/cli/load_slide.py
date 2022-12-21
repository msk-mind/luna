# General imports
import os, json, logging
import click
from pathlib import Path
import yaml

# From common
from luna.common.custom_logger import init_logger
from luna.common.DataStore import DataStore_v2
from luna.common.Node import Node
from luna.common.config import ConfigSet
from luna.common.sparksession import SparkConfig


@click.command()
@click.option(
    "-a",
    "--app_config",
    required=True,
    help="application configuration yaml file. See config.yaml.template for details.",
)
@click.option(
    "-s", "--datastore_id", required=True, help="datastore name. usually a slide id."
)
@click.option(
    "-m",
    "--method_param_path",
    required=True,
    help="json parameter file with path to a WSI delta table.",
)
def cli(app_config, datastore_id, method_param_path):
    """Load a slide to the datastore from the whole slide image table.

    app_config - application configuration yaml file. See config.yaml.template for details.

    datastore_id - datastore name. usually a slide id.

    method_param_path - json parameter file with path to a WSI delta table.

    - job_tag: job tag to use for loading the slide

    - table_path: path to the whole slide image table

    - datastore_path: path to store data
    """
    init_logger()

    with open(method_param_path, "r") as yaml_file:
        method_data = yaml.safe_load(yaml_file)
    load_slide_with_datastore(app_config, datastore_id, method_data)


def load_slide_with_datastore(app_config, datastore_id, method_data):
    """Load a slide to the datastore from the whole slide image table.

    Args:
        app_config (string): path to application configuration file.
        datastore_id (string): datastore name. usually a slide id.
        method_data (dict): method parameters including input, output details.

    Returns:
        None
    """
    logger = logging.getLogger(f"[datastore={datastore_id}]")

    # Do some setup
    cfg = ConfigSet("APP_CFG", config_file=app_config)
    datastore = DataStore_v2(method_data["datastore_path"])
    method_id = method_data["job_tag"]

    # fetch patient_id column
    patient_id_column = method_data.get("patient_id_column_name", None)
    if patient_id_column == "":
        patient_id_column = None

    try:
        spark = SparkConfig().spark_session("APP_CFG", "query_slide")
        slide_id = datastore_id

        if patient_id_column:
            # assumes if patient_id column, source is parquet from dremio
            # right now has nested row-type into dict, todo: account for map type representation of dict in dremio
            df = (
                spark.read.parquet(method_data["table_path"])
                .where(f"slide_id='{slide_id}'")
                .select("path", "metadata", patient_id_column)
                .toPandas()
            )

            if not len(df) == 1:
                print(df)
                raise ValueError(
                    f"Resulting query record is not singular, multiple scan's exist given the container address {slide_id}"
                )

            record = df.loc[0]
            properties = record["metadata"]
            properties["patient_id"] = str(record[patient_id_column])
        else:
            df = (
                spark.read.format("delta")
                .load(method_data["table_path"])
                .where(f"slide_id='{slide_id}'")
                .select("path", "metadata")
                .toPandas()
            )

            if not len(df) == 1:
                print(df)
                raise ValueError(
                    f"Resulting query record is not singular, multiple scan's exist given the container address {slide_id}"
                )

            record = df.loc[0]
            properties = record["metadata"]

        spark.stop()

    except Exception as e:
        logger.exception(f"{e}, stopping job execution...")
        raise e

    # Put results in the data store
    data_path = Path(record["path"].split(":")[-1])
    print(data_path)
    datastore.put(data_path, datastore_id, method_id, "WholeSlideImage", symlink=True)

    with open(
        os.path.join(
            method_data["datastore_path"],
            datastore_id,
            method_id,
            "WholeSlideImage",
            "metadata.json",
        ),
        "w",
    ) as fp:
        json.dump(properties, fp)


if __name__ == "__main__":
    cli()
