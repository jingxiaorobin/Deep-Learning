import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import alexnet


class CNNModel(nn.Module):
    def __init__(self, nclasses=1000):
        super(CNNModel, self).__init__()
        self.alexnet = alexnet(pretrained=False)
        

    def forward(self, x):
        x = self.alexnet(x)
        return x
 
