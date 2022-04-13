# General imports
import os
import logging
import click

import girder_client
import pandas as pd

import json

from geojson import Feature, Point, Polygon, FeatureCollection
from copy import deepcopy

from shapely.geometry import shape
from dask.distributed import Client, as_completed

from luna.common.utils import cli_runner
from luna.common.custom_logger import init_logger
from luna.pathology.dsa.dsa_api_handler import (
    dsa_authenticate,
    get_collection_uuid,
    get_slide_df,
    get_annotation_uuid,
    get_annotation_df,
)

init_logger()
logger = logging.getLogger("dsa_annotation_etl")

_params_ = [
    ("input_dsa_endpoint", str),
    ("collection_name", str),
    ("annotation_name", str),
    ("num_cores", int),
    ("username", str),
    ("password", str),
    ("output_dir", str),
]


@click.command()
@click.argument("input_dsa_endpoint", nargs=1)
@click.option(
    "-o",
    "--output_dir",
    required=False,
    help="path to output directory to save results",
)
@click.option(
    "-c",
    "--collection_name",
    required=False,
    help="name of the collection to pull data from in DSA",
)
@click.option(
    "-a",
    "--annotation_name",
    required=False,
    help="name of the annotations to pull from DSA (same annotation name for all slides)",
)
@click.option(
    "-u",
    "--username",
    required=False,
    help="DSA username, can be inferred from DSA_USERNAME",
)
@click.option(
    "-p",
    "--password",
    required=False,
    help="DSA password, should be inferred from DSA_PASSWORD",
)
@click.option(
    "-nc", "--num_cores", required=False, help="Number of cores to use (default: 4)",
    default=4
)
@click.option(
    "-m",
    "--method_param_path",
    required=False,
    help="path to a metadata json/yaml file with method parameters to reproduce results",
)
def cli(**cli_kwargs):
    """A cli tool

    \b
    Inputs:
        input_dsa_endpoint: Path to the DSA endpoint like http://localhost:8080/dsa/api/v1
    \b
    Outputs:
        slide_annotation_dataset
    \b
    Example:
        export DSA_USERNAME=username
        export DSA_PASSWORD=password
        dsa_annotation_ http://localhost:8080/dsa/api/v1
            --collection_name tcga-data
            --annotation_name TumorVsOther
            -o /data/annotations/
    """
    cli_runner(cli_kwargs, _params_, dsa_annotation_etl)


# Transform imports
def dsa_annotation_etl(
    input_dsa_endpoint,
    username,
    password,
    collection_name,
    annotation_name,
    num_cores,
    output_dir,
):
    """Take

    Args:
        input_dsa_endpoint (str): path to input data
        output_dir (str): output/working directory

    Returns:
        dict: metadata about function call
    """
    girder = girder_client.GirderClient(apiUrl=input_dsa_endpoint)

    dsa_authenticate(girder, username, password)

    collection_uuid = get_collection_uuid(girder, collection_name)

    df_slide_items = get_slide_df(girder, collection_uuid)

    if len(df_slide_items) == 0:
        logger.info("No slides found, exitting!")
        return {}

    # Initialize the DsaAnnotationProcessor
    dap = DsaAnnotationProcessor(girder, annotation_name, output_dir)

    with Client(
        host=os.environ["HOSTNAME"],
        n_workers=((num_cores + 1) // 2),
        threads_per_worker=2,
    ) as client:
        logger.info("Dashboard: " + client.dashboard_link)
        df_polygon_data = pd.concat(
            [
                x.result()
                for x in as_completed(
                    [
                        client.submit(dap.run, row)
                        for _, row in df_slide_items.iterrows()
                    ]
                )
            ]
        )

    # Join the slide level data with the polygon level data, so this is a lot of information!
    df_full_annotation_data = (
        df_slide_items.set_index("slide_item_uuid")
        .join(
            df_polygon_data.set_index("slide_item_uuid"),
            how="right",
            rsuffix="annotation",
        )
        .set_index("slide_id")
    )

    df_full_annotation_data.loc[:, "collection_name"] = collection_name
    df_full_annotation_data.loc[:, "annotation_name"] = annotation_name
    df_full_annotation_data = df_full_annotation_data.drop(columns=["meta"])
    df_full_annotation_data = df_full_annotation_data.rename(
        columns={"group": "group_name"}
    )

    print(df_full_annotation_data)

    # Our dataset is a combination of polyline, point, and geojson annotations!
    logger.info(
        f"""Created {len(df_full_annotation_data.query("type=='geojson'"))} geojsons, {len(df_full_annotation_data.query("type=='point'"))} points, and {len(df_full_annotation_data.query("type=='polyline'"))} polygons"""
    )

    slide_annotation_dataset_path = f"{output_dir}/slide_annotation_dataset_{collection_name}_{annotation_name}.parquet"

    df_full_annotation_data.to_parquet(slide_annotation_dataset_path)

    properties = {
        "slide_annotation_dataset": slide_annotation_dataset_path,
        "segment_keys": {"dsa_collection_uuid": collection_uuid},
    }

    return properties


class DsaAnnotationProcessor:
    def __init__(self, girder, annotation_name, output_dir):
        self.girder = girder
        self.annotation_name = annotation_name
        self.output_dir = output_dir

    def histomics_annotation_table_to_geojson(
        self, df, properties, shape_type_col="type", x_col="x_coords", y_col="y_coords"
    ):
        """Takes a table generated by histomicstk (parse_slide_annotations_into_tables) and creates a geojson"""

        features = []
        df[properties] = df[properties].fillna("None")

        logger.info(f"About to turn {len(df)} geometric annotations into a geojson!")

        for _, row in df.iterrows():
            x, y = deepcopy(row[x_col]), deepcopy(row[y_col])
            if row[shape_type_col] == "polyline":
                x.append(x[0]), y.append(y[0])
                geometry = Polygon(
                    [list(zip(x, y))]
                )  # Polygons are once nested to account for holes

            elif row[shape_type_col] == "point":
                geometry = Point((x[0], y[0]))

            logger.info(f"\tCreated geometry {str(shape(geometry)):.40s}...")
            feature = Feature(
                geometry=geometry, properties={prop: row[prop] for prop in properties}
            )
            features.append(feature)

        feature_collection = FeatureCollection(features)
        logger.info(
            f"Checking geojson, errors with geojson FeatureCollection: {feature_collection.errors()}"
        )

        return feature_collection

    def build_proxy_repr_dsa(self, row):
        """Build a proxy table slice given, primarily, a DSA itemId (slide_item_uuid)"""

        itemId = row.slide_item_uuid
        slide_id = row.slide_id

        logger.info(
            f"Trying to process annotation for slide_id={slide_id}, item_id={itemId}"
        )

        annotation_uuid = get_annotation_uuid(
            self.girder, item_id=itemId, annotation_name=self.annotation_name
        )

        if annotation_uuid is None:
            return None

        df_annotations = get_annotation_df(self.girder, annotation_uuid)

        print(df_annotations)

        # This turns the regional data into a nice geojson
        feature_collection = self.histomics_annotation_table_to_geojson(
            df_annotations,
            ["annotation_girder_id", "element_girder_id", "group", "label"],
            shape_type_col="type",
            x_col="x_coords",
            y_col="y_coords",
        )

        slide_geojson_path = f"{self.output_dir}/{slide_id}.annotation.geojson"
        with open(slide_geojson_path, "w") as fp:
            json.dump(feature_collection, fp)  # Finally, save it!

        df_annotation_proxy = pd.concat(
            [
                df_annotations,
                pd.DataFrame(
                    [
                        {
                            "slide_item_uuid": itemId,
                            "type": "geojson",
                            "slide_geojson": slide_geojson_path,
                        }
                    ]
                ),
            ]
        )  # Add our geojson as a special type of annotation

        return df_annotation_proxy

    def run(self, row):
        """Run DsaAnnotationProcessor

        Args:
            row (string): row of a DSA slide table

        Returns:
            pd.DataFrame: annotation metadata
        """

        df = self.build_proxy_repr_dsa(row)

        return df


if __name__ == "__main__":
    cli(auto_envvar_prefix="DSA")
