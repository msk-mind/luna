"""load_classifier
Modified version of [1], rewritten to load tissue classifier models trained by
the MIND team.
[1] https://github.com/msk-mind/data-processing/blob/refactor-cli-fix-tests/data_processing/pathology/models/tissuenet.py
"""
import torch
import torchvision
import torchvision.models as models


def get_transform():
    """ Transformer which generates a torch tensor compatible with the model """
    return torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
    ])


def get_classifier(
    checkpoint_path="/gpfs/mskmindhdp_emc/user/shared_data_folder/kohlia/tile_classifier/ckpts/1.ckpt",
    activation=None,
    n_classes=5,
):
    """ Return model given checkpoint_path """
    model = models.resnet18(num_classes=n_classes)
    model.load_state_dict(
        torch.load(checkpoint_path, map_location="cpu")["model_states"]["net"]
    )

    return model