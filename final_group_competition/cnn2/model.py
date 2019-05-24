import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


class CNNModel(nn.Module):
    def __init__(self, nclasses=1000):
        super(CNNModel, self).__init__()
        self.resnet = resnet18()
        self.resnet.fc = nn.Linear(2048, nclasses)
        
    def forward(self, x):
        return self.resnet(x)
