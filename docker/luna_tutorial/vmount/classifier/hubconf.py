import os 
from collections import OrderedDict
import torch
from torch import nn

from luna.pathology.analysis.ml import TorchTransformModel
from luna.common.utils import load_func



class TutorialModel(TorchTransformModel):
    preprocess = nn.Identity()

    def __init__(self, network, num_labels):
        self.checkpoint = os.path.join(os.path.dirname(__file__), "model.checkpoint")
        self.state_dict_dp = torch.load(self.checkpoint)['model_state_dict']
        self.network = load_func(network)
        self.model = load_func(network)(num_classes=num_labels)

        # removing data parallel formatting from weights
        self.state_dict = OrderedDict()
        for k, v in self.state_dict_dp.items():
            name = k[7:] # remove `module.`
            self.state_dict[name] = v
        
        self.model.load_state_dict(self.state_dict)

    def get_preprocess(self):
        return self.preprocess

    def transform(self, X):
        # equivalent of ToTensor()
        X = X.permute(0, 3, 1, 2).float() / 255
        out = self.model(X).view(X.shape[0], -1).cpu().numpy()
        return out




def tissue_classifier(**kwargs):
    return TutorialModel(**kwargs)
