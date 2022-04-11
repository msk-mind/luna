import torch
from torch import nn

from luna.pathology.analysis.ml import TorchTransformModel


class MyCustomModel(TorchTransformModel):
    preprocess = nn.Identity()

    def __init__(self, n_channels):
        self.model = nn.Sequential(
            nn.Conv2d(3, n_channels, kernel_size=3), nn.AdaptiveAvgPool2d((1, 1))
        )

        if n_channels == 2:
            self.column_labels = {0: "Background", 1: "Tumor"}

    def get_preprocess(self):
        return self.preprocess

    def transform(self, X):
        X = X.permute(0, 3, 1, 2).float() / 255
        out = self.model(X).view(X.shape[0], -1).cpu().numpy()
        return out


class Resnet(TorchTransformModel):
    preprocess = nn.Identity()

    def __init__(self, depth, pretrained):
        # del kwargs['depth']
        self.model = torch.hub.load(
            "pytorch/vision:v0.10.0", f"resnet{depth}", pretrained=pretrained
        )

    def get_preprocess(self):
        return self.preprocess

    def transform(self, X):
        X = X.permute(0, 3, 1, 2).float() / 255
        out = self.model(X).view(X.shape[0], -1).cpu().numpy()
        return out


def test_custom_model(n_channels=2):
    return MyCustomModel(n_channels)


def test_resnet(**kwargs):
    return Resnet(**kwargs)
