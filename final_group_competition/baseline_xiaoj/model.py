import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50


class CNNModel(nn.Module):
    def __init__(self, nclasses=1000):
        super(CNNModel, self).__init__()
        self.resnet = resnet50(pretrained=False)
        self.resnet.fc = nn.Sequential(
            nn.Linear(2048, 1024, bias=False), 
            nn.Linear(1024, nclasses, bias=False))

    def forward(self, x):
        x = self.resnet(x)
        return x
 
