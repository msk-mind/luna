import torch
from torch import nn

from luna.pathology.common.ml import TorchTransformModel

class MyModel(TorchTransformModel):
    preprocess = nn.Identity()

    def __init__(self, n_channels):
        self.model = nn.Sequential(
            nn.Conv2d(3, n_channels, kernel_size=3),
            nn.AdaptiveAvgPool2d((1,1))
        )

    def get_preprocess(self):
        return self.preprocess

    def transform(self, X):
        X = X.permute(0, 3, 1, 2).float()
        out = self.model(X).view(X.shape[0], -1)
        return out

def testmodel(n_channels=8): return MyModel(n_channels)
