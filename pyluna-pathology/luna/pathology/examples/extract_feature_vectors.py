import os, logging, click
import numpy as np
import time
import h5py
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from CLAM.datasets.dataset_h5 import Dataset_All_Bags, Whole_Slide_Bag_FP, eval_transforms
from CLAM.models.resnet_custom import resnet50_baseline
from CLAM.utils.utils import print_network, collate_features
from CLAM.utils.file_utils import save_hdf5

from luna.common.custom_logger   import init_logger
from luna.common.utils import cli_runner
from luna.pathology.common.utils import address_to_coord

init_logger()
logger = logging.getLogger('extract_feature_vectors')

_params_ = [('csv_path', str), ('output_dir', str), ('batch_size', int), ('no_auto_skip', bool)]

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class Luna_Bag(Dataset):
    """
    Dataset with Luna tile images.
    Luna tile images are saved in h5 file with address-tile image array as key-value pairs.
    """
    def __init__(self,
        file_path,
        pretrained=False,
        custom_transforms=None):
        """
        Args:
            file_path (string): Path to the .h5 file containing patched data.
            pretrained (bool): Use ImageNet transforms
            custom_transforms (callable, optional): Optional transform to be applied on a sample
        """
        self.pretrained = pretrained
        if not custom_transforms:
            self.roi_transforms = eval_transforms(pretrained=pretrained)
        else:
            self.roi_transforms = custom_transforms

        self.file_path = file_path

        with h5py.File(self.file_path, "r") as f:
            dset = list(f.keys())
            self.length = len(dset)
            
        self.summary()
            
    def __len__(self):
        return self.length

    def summary(self):
        logger.info('\nfeature extraction settings')
        logger.info(f'pretrained: {self.pretrained}')
        logger.info(f'transformations: {self.roi_transforms}')

    def __getitem__(self, idx):
        with h5py.File(self.file_path,'r') as hdf5_file:
            addr = list(hdf5_file.keys())[idx]
            img = np.array(hdf5_file[addr])
            img = Image.fromarray(img)
            img = self.roi_transforms(img).unsqueeze(0)
        return img, address_to_coord(addr)
    
    
def compute_w_loader(file_path, output_path, model,
     batch_size = 8, verbose = 0, print_every=20, pretrained=True):
    """
    Extract features and save the vectors.

    Args:
        file_path: directory of bag (.h5 file)
        output_path: directory to save computed features (.h5 file)
        model: pytorch model
        batch_size: batch_size for computing features in batches
        verbose: level of feedback
        pretrained: use weights pretrained on imagenet
    """
    dataset = Luna_Bag(file_path=file_path,
                          pretrained=pretrained)

    kwargs = {'num_workers': 4, 'pin_memory': True} if device.type == "cuda" else {}
    loader = DataLoader(dataset=dataset, batch_size=batch_size, **kwargs, collate_fn=collate_features)

    if verbose > 0:
        logger.info('processing {}: total of {} batches'.format(file_path,len(loader)))

    mode = 'w'
    for count, (batch, coords) in enumerate(loader):
        with torch.no_grad():    
            if count % print_every == 0:
                logger.info('batch {}/{}, {} files processed'.format(count, len(loader), count * batch_size))
            batch = batch.to(device, non_blocking=True)
            
            features = model(batch)
            features = features.cpu().numpy()

            asset_dict = {'features': features, 'coords': coords}
            save_hdf5(output_path, asset_dict, attr_dict= None, mode=mode)
            mode = 'a'
    
    return output_path

@click.command()
@click.option('-c', '--csv_path', type=str, required=False,
              help='path to csv with slide_id, tile_image_file columns')
@click.option('-o', '--output_dir', type=str, required=False,
              help='path to save extracted features')
@click.option('-bs', '--batch_size', type=int, default=256)
@click.option('-as', '--no_auto_skip', default=False, is_flag=True,
              help='If true, override existing output. By default, skip if output exists.')
@click.option('-m', '--method_param_path', required=False,
              help='path to a metadata json/yaml file with method parameters to reproduce results')
def cli(**cli_kwargs):
    """An example showing how Luna tile images can be used with CLAM.
    Extract 1024-dimensional feature vector from each tile, using a pre-trained ResNet50.

    Note:
        Adapted from feature extraction CLI: https://github.com/msk-mind/CLAM/blob/master/extract_features_fp.py

    Setup (utilizes CLAM):
        pip install pyluna[pathology]
        git clone https://github.com/msk-mind/CLAM.git
        export PYTHONPATH=$PYTONPATH:/path/to/CLAM:.

    Example:
        !python3 -m luna.pathology.examples.extract_feature_vectors \
            --csv_path dataset.csv \
            --output_dir /path/to/output \
    """
    cli_runner(cli_kwargs, _params_, extract_feature_vectors)


def extract_feature_vectors(csv_path, output_dir, batch_size, no_auto_skip):
    """
    Extract 1024-dimensional feature vector from each tile, using a pre-trained ResNet50.

    Args:
        csv_path (str): path to csv with slide_id, tile_image_file columns
        output_dir (str): path to save extracted features
        batch_size (int): batch size
        no_auto_skip (bool): If true, override existing output. By default, skip if output exists.

    Returns:
        dict: metadata about function call
    """
    if csv_path is None:
        raise NotImplementedError

    bags_dataset = Dataset_All_Bags(csv_path)
    
    os.makedirs(os.path.join(output_dir, 'pt_files'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'h5_files'), exist_ok=True)
    dest_files = os.listdir(os.path.join(output_dir, 'pt_files'))

    print('loading model checkpoint')
    model = resnet50_baseline(pretrained=True)
    model = model.to(device)
    
    # print_network(model)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        
    model.eval()
    total = len(bags_dataset)

    for bag_candidate_idx in range(total):
        slide_id, h5_file_path = bags_dataset[bag_candidate_idx]

        bag_name = slide_id+'.h5'
        logger.info('\nprogress: {}/{}'.format(bag_candidate_idx+1, total))
        logger.info(f'processing {slide_id}')
        if not no_auto_skip and slide_id+'.pt' in dest_files:
            logger.info('skipped {}'.format(slide_id))
            continue 

        output_path = os.path.join(output_dir, 'h5_files', bag_name)
        time_start = time.time()
        output_file_path = compute_w_loader(h5_file_path, 
                                            output_path, 
                                            model = model, 
                                            batch_size = batch_size,
                                            verbose = 1, 
                                            print_every = 20)
        time_elapsed = time.time() - time_start
        logger.info('\ncomputing features for {} took {} s'.format(output_file_path, time_elapsed))
        file = h5py.File(output_file_path, "r")

        features = file['features'][:]
        logger.info(f'features size: {features.shape}')
        logger.info(f"coordinates size: {file['coords'].shape}")
        features = torch.from_numpy(features)
        bag_base, _ = os.path.splitext(bag_name)
        torch.save(features, os.path.join(output_dir, 'pt_files', bag_base+'.pt'))

    properties = {
        "dataset_csv": csv_path,
        "feat_dir": output_dir,
        "batch_size": batch_size
    }

    return properties

if __name__ == '__main__':
    cli()
