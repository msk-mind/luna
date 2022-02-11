# General imports
import os, json, logging, yaml
import click

from luna.common.custom_logger   import init_logger

init_logger()
logger = logging.getLogger("dsa_annotation_etl") ### Add CLI tool name

from luna.common.utils import cli_runner

_params_ = [('input_dsa_endpoint', str), ('collection_name', str), ('annotation_name', str), ('num_cores', int), ('username', str), ('password', str), ('output_dir', str)]

@click.command()
@click.argument('input_dsa_endpoint', nargs=1)
@click.option('-o', '--output_dir', required=False,
              help='path to output directory to save results')
@click.option('-c', '--collection-name', required=False,
              help='path to output directory to save results')
@click.option('-a', '--annotation-name', required=False,
              help='path to output directory to save results')
@click.option('-u', '--username', required=False,
              help='path to output directory to save results')
@click.option('-nc', '--num_cores', required=False,
              help='Number of cores to use', default=4)
@click.option('-p', '--password', required=False,
              help='path to output directory to save results')

### Additional options
@click.option('-m', '--method_param_path', required=False,
              help='path to a metadata json/yaml file with method parameters to reproduce results')
def cli(**cli_kwargs):
    """ A cli tool

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
    cli_runner( cli_kwargs, _params_, dsa_annotation_etl)

import girder_client
import pandas as pd
from pathlib import Path

from dask.distributed import Client, as_completed
from tqdm import tqdm 

### Transform imports 
def dsa_annotation_etl(input_dsa_endpoint, username, password, collection_name, annotation_name, num_cores, output_dir):
    """ CLI tool method

    Args:
        input_dsa_endpoint (str): path to input data
        output_dir (str): output/working directory

    Returns:
        dict: metadata about function call
    """
    girder = girder_client.GirderClient(apiUrl=input_dsa_endpoint)

    girder.authenticate(username, password)
    
    try:
        collections = pd.DataFrame(girder.listCollection()).set_index('_id')
        logger.info(f"Connected to DSA @ {input_dsa_endpoint}")
    except Exception as exc:
        logger.error("Couldn't connect to DSA: {exc}")

   
    logger.info(f"Found collections {collections.to_string(max_colwidth=100)}")

    q_collection_result = collections.query( f"name=='{collection_name}'" )

    if not len(q_collection_result)==1:
        raise RuntimeError(f"No matching collection '{collection_name}'")
        
    collection_id = q_collection_result.index.item()

    q_slide_items = pd.DataFrame(girder.listResource(f'resource/{collection_id}/items',{'type':'collection'})).dropna(subset=['largeImage'])
    
    q_slide_items['slide_id'] =      q_slide_items['name'].apply(lambda x: Path(x).stem)
    q_slide_items['slide_item_id'] = q_slide_items['_id']

    logger.info(f"Found {len(q_slide_items)} slides!")

    dap = DsaAnnotationProcessor(girder, annotation_name, output_dir)

    with Client(host=os.environ['HOSTNAME'], n_workers=((num_cores + 1) // 2), threads_per_worker=2) as client:
        logger.info ("Dashboard: "+ client.dashboard_link)
        df_polygon_data = pd.concat([x.result() for x in as_completed([client.submit(dap.run, row) for _, row in q_slide_items.iterrows()])])
    
    df_annotation_data = q_slide_items.set_index('slide_item_id').join(df_polygon_data.set_index('slide_item_id'), how='right', rsuffix='annotation').set_index('slide_id')
        
    print (set(df_annotation_data.columns))
    print (df_annotation_data)

    df_annotation_data.to_csv(f"{output_dir}/annotation_data.csv")

    properties = {

    }

    return properties


import json

from geojson import Feature, Point, Polygon, FeatureCollection
from copy import deepcopy
import histomicstk
import histomicstk.annotations_and_masks.annotation_and_mask_utils
class DsaAnnotationProcessor:
    def __init__(self, girder, annotation_name, output_dir):
        self.girder  = girder
        self.annotation_name = annotation_name
        self.output_dir = output_dir

    def histomics_annotation_table_to_geojson(self, df, properties, shape_type_col='type', x_col='x_coords', y_col='y_coords'):
        """ Takes a table generated by histomicstk (parse_slide_annotations_into_tables) and creates a geojson """
        features = []
        df[properties] = df[properties].fillna('None')
            
        for _, row in df.iterrows():
            x, y = deepcopy(row[x_col]), deepcopy(row[y_col])
            if row[shape_type_col] == 'polyline':
                x.append(x[0]), y.append(y[0])
                geometry = Polygon([list(zip(x,y))])

            elif row[shape_type_col] == 'point':
                geometry = Point((x[0], y[0]))
                                            
            feature = Feature(geometry=geometry, properties={prop:row[prop] for prop in properties})
            features.append(feature)
                        
        feature_collection = FeatureCollection(features)
        print ("Checking geojson, errors with geojson FeatureCollection:", feature_collection.errors())
        return feature_collection

    def build_proxy_repr_dsa(self, row):
        """ Build a proxy table slice given, primarily, a DSA itemId (slide_item_id)"""
        
        itemId   = row.slide_item_id
        slide_id = row.slide_id

        df = pd.DataFrame(self.girder.get( f"annotation?itemId={itemId}" )).set_index('_id')
        q_annotation_result = df.join(df['annotation'].apply(pd.Series)).query( f"name=='{self.annotation_name}'")
        
        if not len(q_annotation_result)==1:
            logger.info(f"No matching annotation '{self.annotation_name}' in slide {slide_id}")
            return None
        
        annotation_id = q_annotation_result.reset_index()['_id'].item()

        annot = self.girder.get( f"annotation/{annotation_id}" )
        df_summary, df_regions = histomicstk.annotations_and_masks.annotation_and_mask_utils.parse_slide_annotations_into_tables([annot])
        df_regions['x_coords'] = [ [int(x) for x in coords_x.split(',') ] for coords_x in df_regions['coords_x'] ]
        df_regions['y_coords'] = [ [int(x) for x in coords_x.split(',') ] for coords_x in df_regions['coords_y'] ]
        
        df_annotation_proxy = df_summary.set_index("annotation_girder_id").join(df_regions.set_index("annotation_girder_id")).reset_index()

        df_annotation_proxy['slide_item_id'] = itemId
        
        feature_collection = self.histomics_annotation_table_to_geojson(df_annotation_proxy, ['annotation_girder_id', 'element_girder_id', 'group', 'label'], shape_type_col='type', x_col='x_coords', y_col='y_coords')
        
        slide_geojson_path = f"{self.output_dir}/{slide_id}.annotation.geojson"
        with open(slide_geojson_path, 'w') as fp:
            json.dump(feature_collection, fp)

        df_annotation_proxy = pd.concat([df_annotation_proxy, pd.DataFrame([{'slide_item_id':itemId, 'type': 'geojson', 'slide_geojson':slide_geojson_path}])])
        
        return df_annotation_proxy

    def run(self, row):
        """ Run DsaAnnotationProcessor

        Args:
            row (string): row of a DSA slide table

        Returns:
            pd.DataFrame: annotation metadata
        """

        df = self.build_proxy_repr_dsa(row)

        return df


if __name__ == "__main__":
    cli(auto_envvar_prefix='DSA')
