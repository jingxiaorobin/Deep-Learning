import torch 
import torch.nn as nn
import torch.nn.functional as F

class CAEClassifier(nn.Module):
    def __init__(self, fc_input_features, n_classes):
        super(CAEClassifier, self).__init__()
        self.fc_input_features = fc_input_features
        self.n_classes = n_classes
        self.conv1 = nn.Conv2d(3,16,5,1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        
        self.conv2 = nn.Conv2d(16,32,5,1)
        self.bn2 = nn.BatchNorm2d(32)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        
        self.conv3 = nn.Conv2d(32,64,4,1)
        self.bn3 = nn.BatchNorm2d(64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=1, return_indices=True)
        
        # 17*17*64
        self.fc1 = nn.Linear(self.fc_input_features, 8192)

    # def initialize_from_sae(self, cae):
    #     # weight
    #     self.conv1.weight = cae.conv1.weight
    #     self.conv2.weight = cae.conv2.weight
    #     self.conv3.weight = cae.conv3.weight
    #     self.bn1.weight = cae.bn1.weight
    #     self.bn2.weight = cae.bn2.weight
    #     self.bn3.weight = cae.bn3.weight
    #     self.fc1.weight = cae.fc1.weight

    #     # bias
    #     self.conv1.bias = cae.conv1.bias
    #     self.conv2.bias = cae.conv2.bias
    #     self.conv3.bias = cae.conv3.bias
    #     self.bn1.bias = cae.bn1.bias
    #     self.bn2.bias = cae.bn2.bias
    #     self.bn3.bias = cae.bn3.bias
    #     self.fc1.bias = cae.fc1.bias


    #     # this can work now
    #     self.conv1.weight.requires_grad = False
    #     self.conv1.weight.requires_grad = False
    #     self.conv1.weight.requires_grad = False
    def initialize(self):
        self.fc2 = nn.Linear(8192, 4096)
        self.fc3 = nn.Linear(4096, self.n_classes)

    def forward(self, x):
        # encode
        x = self.relu(self.bn1(self.conv1(x)))
        x, poolIdx1 = self.maxpool1(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x, poolIdx2 = self.maxpool2(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x, poolIdx3 = self.maxpool3(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        return F.log_softmax(x, dim=1)
        
