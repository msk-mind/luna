# General imports
import os, json, logging
import click

from luna.common.custom_logger   import init_logger

init_logger()
logger = logging.getLogger('infer_tille_labels')

import torch
from torch.utils.data import DataLoader
from luna.pathology.common.ml import BaseTorchTileDataset, BaseTorchTileClassifier

import pandas as pd
from tqdm import tqdm


class TileDatasetGithub(BaseTorchTileDataset):
    def setup(self, repo_name, transform_name):
        self.transform = torch.hub.load(repo_name, transform_name)
    def preprocess(self, input_tile):
        return self.transform(input_tile)
    
class TileClassifierGithub(BaseTorchTileClassifier):
    def setup(self, repo_name, model_name, weight_tag):
        self.model = torch.hub.load(repo_name, model_name, weight_tag=weight_tag)
    def predict(self, input_tiles):
        return self.model(input_tiles)

@click.command()
@click.option('-i', '--input_data', required=True,
              help='path to input data')
@click.option('-o', '--output_dir', required=True,
              help='path to output directory to save results')
@click.option('-r', '--repo_name', required=True,
              help="repository name to pull model and weight from, e.g. msk-mind/luna-ml")
@click.option('-t', '--transform_name', required=True,
              help="torch hub transform name")   
@click.option('-m', '--model_name', required=True,
              help="torch hub model name")    
@click.option('-w', '--weight_tag', required=True,
              help="weight tag filename")    

def cli(input_data, output_dir, repo_name, transform_name, model_name, weight_tag):
    """
    infer_tile_labels
        -i /gpfs/mskmindhdp_emc/data/CRC_21-167/tables/tiles/1412934/crc_generate_all_tiles/TileImages/data/ 
        -o ./tmp/1412934 -r msk-mind/luna-ml:main 
        -t tissue_tile_net_transform 
        -m tissue_tile_net_model_5_class 
        -w main:tissue_net_2021-01-19_21.05.24-e17.pth
    """
    logger.info(f"Running infer_tille_labels, input_data={input_data}, output_dir={output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Loading model and transform:, repo_name={repo_name}, transform_name={transform_name}, model_name={model_name}")
    logger.info(f"Using weights weight_tag={weight_tag}")
    tile_dataset     = TileDatasetGithub(tile_path=input_data, repo_name=repo_name, transform_name=transform_name)
    tile_classifier  = TileClassifierGithub(repo_name=repo_name, model_name=model_name, weight_tag=weight_tag)

    tile_loader = DataLoader(
            tile_dataset,
            num_workers=24,
            batch_size=256,
            pin_memory=True
        )

    with torch.no_grad():
        df_scores = pd.concat([tile_classifier(index, data) for index, data in tqdm(tile_loader)])
        
    if hasattr(tile_classifier.model, 'class_labels'):
        logger.info(f"Mapping column labels -> {tile_classifier.model.class_labels}")
        df_scores = df_scores.rename(columns=tile_classifier.model.class_labels)

    df_output = tile_dataset.tile_manifest.join(df_scores)    

    logger.info(df_output)

    output_file = os.path.join(output_dir, "tile_scores_and_labels_pytorch_inference.csv")
    df_output.to_csv(output_file)

    properties = {
        "data": output_file,
        "total_tiles": len(df_output),
        "available_labels": list(df_output.columns)
    }

    with open(os.path.join(output_dir, "metadata.json"), "w") as fp:
        json.dump(properties, fp)

if __name__ == "__main__":
    cli()
