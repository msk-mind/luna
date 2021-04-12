"""load_classifier
Modified version of [1], rewritten to load tissue classifier models trained by
the MIND team.
[1] https://github.com/msk-mind/data-processing/blob/refactor-cli-fix-tests/data_processing/pathology/models/tissuenet.py
"""
import torch
import torchvision
from torchvision.models import resnet18


class EngTissueTileNet(torch.nn.Module):
    def __init__(self, model, n_classes, activation=None):
        super(EngTissueTileNet, self).__init__()
        if type(model) in [torchvision.models.resnet.ResNet]:
            model.fc = torch.nn.Linear(512, n_classes)
        else:
            raise NotImplementedError

        self.model = model
        self.activation = activation

    def forward(self, x):
        y = self.model(x)

        if self.activation:
            y = self.activation(y)

        return y


def get_transform():
    """ Transformer which generates a torch tensor compatible with the model """
    return torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
        ]
    )


def get_classifier(
    checkpoint_path="/gpfs/mskmind_ess/kohlia/tile_classifier/ckpts/1.ckpt",
    activation=None,
    n_classes=5,
):
    """ Return model given checkpoint_path """
    model = EngTissueTileNet(resnet18(), n_classes, activation=activation)
    model.load_state_dict(
        torch.load(checkpoint_path["model_states"]["net"], map_location="cpu")
    )
    return model