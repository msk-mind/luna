"""load_classifier
Modified version of [1], rewritten to load tissue classifier models trained by
the MIND team.
[1] https://github.com/msk-mind/data-processing/blob/refactor-cli-fix-tests/data_processing.pathology/models/tissuenet.py
"""
from typing import Callable
import torch
import torchvision
import torchvision.models as models


def get_transform() -> Callable:
    """get_transform

    transformer which generates a torch tensor compatible with the model 
    
    Args: 
        none
    Returns
        torchvision.Transform: transform object for tensor conversion

    """
    return torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
    ])


def get_classifier(
        checkpoint_path:str="/gpfs/mskmindhdp_emc/user/shared_data_folder/kohlia/tile_classifier/ckpts/1.ckpt",
    n_classes:int=5,
):
    """get_classifier

    loads a model from a checkpoint and unpacks the network 
    
    Args:
        checkpoint_path (str): path to model checkpoint
        n_classes: number of classes used in training, used to set final layer in
            PyTorch model
    Returns:
        nn.Module: PyTorch module with loaded weights 
    """
    model = models.resnet18(num_classes=n_classes)
    model.load_state_dict(
        torch.load(checkpoint_path, map_location="cpu")["model_states"]["net"]
    )

    return model
