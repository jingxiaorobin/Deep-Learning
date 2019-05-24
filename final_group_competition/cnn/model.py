import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50


class CNNModel(nn.Module):
    def __init__(self, nclasses=1000):
        super(CNNModel, self).__init__()
        self.resnet = resnet50()

    def forward(self, x):
        return self.resnet(x)
