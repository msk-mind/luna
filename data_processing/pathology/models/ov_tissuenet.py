import torch
import torchvision
from torchvision.models import resnet18, resnet34, resnet50, squeezenet1_1, vgg19_bn

class TissueTileNet(torch.nn.Module):
    def __init__(self, model, n_classes, activation=None):
        super(TissueTileNet, self).__init__()
        if type(model) in [torchvision.models.resnet.ResNet]:
            model.fc = torch.nn.Linear(512, n_classes)
        elif type(model) == torchvision.models.squeezenet.SqueezeNet:
            list(model.children())[1][1] = torch.nn.Conv2d(512, n_classes, kernel_size=1, stride=1)
        else:
            raise NotImplementedError
        self.model = model
        self.activation = activation

    def forward(self, x):
        y = self.model(x)

        if self.activation:
            y = self.activation(y)

        return y

def get_transform ():
    """ Transformer which generates a torch tensor compatible with the model """
    return torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(), 
        torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

def get_classifier (checkpoint_path='/gpfs/mskmind_ess/boehmk/histocox/checkpoints/2021-01-19_21.05.24_fold-2_epoch017.torch', activation=None, n_classes=4):
    """ Return model given checkpoint_path """
    model = TissueTileNet(resnet18(), n_classes, activation=activation)
    model.load_state_dict(torch.load(
        checkpoint_path,
        map_location='cpu')
    )
    return model
