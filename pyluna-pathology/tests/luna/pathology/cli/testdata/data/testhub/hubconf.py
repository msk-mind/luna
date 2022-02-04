import torch
from torch import nn
from  torchvision import transforms

from luna.pathology.common.ml import TorchTransformModel

class MyModel(TorchTransformModel):
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.ConvertImageDtype(torch.float)
    ])

    model = nn.Sequential(
        nn.Conv2d(3, 8, kernel_size=3),
        nn.AdaptiveAvgPool2d((1,1))
    )

    def get_preprocess(self):
        return self.preprocess

    def transform(self, X):
        out = self.model(X).view(X.shape[0], -1)
        return out.cpu().numpy()

def testmodel(): return MyModel()
