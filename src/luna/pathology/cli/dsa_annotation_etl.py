# imports
import json
from copy import deepcopy
from pathlib import Path

import fire
import fsspec  # type: ignore
import girder_client
import pandas as pd
import requests
from dask.distributed import as_completed
from geojson import Feature, FeatureCollection, Point, Polygon
from loguru import logger
from shapely.geometry import shape

from luna.common.dask import get_or_create_dask_client
from luna.common.utils import get_config, save_metadata, timed
from luna.pathology.dsa.dsa_api_handler import (
    get_annotation_df,
    get_annotation_uuid,
    get_collection_uuid,
    get_slide_df,
    system_check,
)


@timed
@save_metadata
def cli(
    dsa_endpoint: str = "???",
    collection_name: str = "???",
    annotation_name: str = "???",
    username: str = "${oc.env:DSA_USERNAME}",
    password: str = "${oc.env:DSA_PASSWORD}",
    local_config: str = "",
    output_urlpath: str = ".",
    storage_options: dict = {},
):
    """DSA annotation ETL
    Args:
        dsa_endpoint (str): path to input data
        collection_name (str): collection name in DSA
        annotation_name (str): annotation name
        username (str): DSA username (defaults to environment variable DSA_USERNAME)
        password (str): DSA password (defaults to environment variable DSA_PASSWORD)
        local_config (str): local config yaml url/path
        output_urlpath (str): output/working url/path prefix
        storage_options (dict): options to pass to reading/writing functions

    Returns:
        pd.DataFrame: metadata from function call
    """
    config = get_config(vars())

    df_full_annotation_data = dsa_annotation_etl(
        config["dsa_endpoint"],
        config["collection_name"],
        config["annotation_name"],
        config["username"],
        config["password"],
        config["output_urlpath"],
        config["storage_options"],
    )

    output_fs, output_path = fsspec.core.url_to_fs(
        config["output_urlpath"], **config["storage_options"]
    )

    slide_annotation_dataset_path = str(
        Path(output_path)
        / f"slide_annotation_dataset_{config['collection_name']}_{config['annotation_name']}.parquet"
    )

    if len(df_full_annotation_data) > 0:
        with output_fs.open(slide_annotation_dataset_path, "wb") as of:
            df_full_annotation_data.to_parquet(of)

        properties = {
            "slide_annotation_dataset": slide_annotation_dataset_path,
            "segment_keys": {
                "dsa_collection_uuid": df_full_annotation_data["collection_uuid"][0]
            },
        }
        return properties


# Transform imports
def dsa_annotation_etl(
    dsa_endpoint: str,
    collection_name: str,
    annotation_name: str,
    username: str,
    password: str,
    output_urlpath: str,
    storage_options: dict,
):
    """DSA annotation ETL

    Args:
        dsa_endpoint (str): path to input data
        collection_name (str): collection name in DSA
        annotation_name (str): annotation name
        username (str): DSA username
        password (str): DSA password
        output_urlpath (str): output/working url/path prefix
        storage_options (dict): options to pass to reading/writing functions

    Returns:
        pd.DataFrame: slide etl dataframe with annotation columns
    """
    client = get_or_create_dask_client()
    # girder = girder_client.GirderClient(apiUrl=dsa_endpoint)
    try:
        girder = girder_client.GirderClient(apiUrl=dsa_endpoint)
        # girder python client doesn't support turning off ssl verify.
        # can be removed once we replace the self-signed cert
        session = requests.Session()
        session.verify = False
        girder._session = session
        girder.authenticate(username, password)

        # check DSA connection
        system_check(girder)

    except Exception as exc:
        logger.error(exc)
        raise RuntimeError("Error connecting to DSA API")

    # dsa_authenticate(girder, username, password)

    collection_uuid = get_collection_uuid(girder, collection_name)

    df_slide_items = get_slide_df(girder, collection_uuid)

    if len(df_slide_items) == 0:
        logger.info("No slides found, exitting!")
        return {}

    # Initialize the DsaAnnotationProcessor
    dap = DsaAnnotationProcessor(
        girder, annotation_name, output_urlpath, storage_options
    )

    logger.info("Dashboard: " + client.dashboard_link)
    df_polygon_data = pd.concat(
        [
            x.result()
            for x in as_completed(
                [client.submit(dap.run, row) for _, row in df_slide_items.iterrows()]
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

    df_full_annotation_data.loc[:, "collection_uuid"] = collection_uuid
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

    return df_full_annotation_data


class DsaAnnotationProcessor:
    def __init__(self, girder, annotation_name, output_urlpath, storage_options):
        self.girder = girder
        self.annotation_name = annotation_name
        self.output_urlpath = output_urlpath
        self.storage_options = storage_options

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
            else:
                continue  # don't process non-polyline(regional) or point annotations

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

        annotation_uuids = get_annotation_uuid(
            self.girder, item_id=itemId, annotation_name=self.annotation_name
        )

        if annotation_uuids is None:
            return None

        # need to loop through annotation uuids since the same annotation name
        # can coorespond to multiple uuids (a 'Regional' annotation on the same
        # slide made two days apart)
        df_annotations = []
        for annotation_uuid in annotation_uuids:
            df_annotation = get_annotation_df(self.girder, annotation_uuid)
            df_annotations.append(df_annotation)

        df_annotations = pd.concat(df_annotations)

        # This turns the regional data into a nice geojson
        feature_collection = self.histomics_annotation_table_to_geojson(
            df_annotations,
            ["annotation_girder_id", "element_girder_id", "group", "label"],
            shape_type_col="type",
            x_col="x_coords",
            y_col="y_coords",
        )

        fs, urlpath = fsspec.core.url_to_fs(self.output_urlpath, **self.storage_options)

        slide_geojson_path = str(Path(urlpath) / f"{slide_id}.annotation.geojson")
        with fs.open(slide_geojson_path, "w") as fp:
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


def fire_cli():
    fire.Fire(cli)


if __name__ == "__main__":
    fire_cli()
