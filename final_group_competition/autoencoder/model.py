import torch
import torch.nn as nn
import torch.functional as F
import numpy as np

# need to modify
class ConvolutionalAutoEncoder(nn.Module):
    def __init__(self, fc_input_features):
        super(ConvolutionalAutoEncoder, self).__init__()
        self.fc_input_features = fc_input_features
        
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
#         self.fc2 = nn.Linear(16384, 8192)
#         self.fc3 = nn.Linear(8192, 16384)
        self.fc2 = nn.Linear(8192, self.fc_input_features)


        self.unpool1 = nn.MaxUnpool2d(kernel_size=2, stride=1)
        self.unpool2 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.unpool3 = nn.MaxUnpool2d(kernel_size=2, stride=2)
    
        self.deconv1 = nn.ConvTranspose2d(64,32,4,1)
        self.deconv2 = nn.ConvTranspose2d(32,16,5,1)
        self.deconv3 = nn.ConvTranspose2d(16,3,5,1)

    def forward(self, x):
        # encode
        x = self.relu(self.bn1(self.conv1(x)))
        x, poolIdx1 = self.maxpool1(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x, poolIdx2 = self.maxpool2(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x, poolIdx3 = self.maxpool3(x)
        x = x.view(x.size(0), -1)
        encoded = self.relu(self.fc1(x))
#         encoded = self.relu(self.fc2(x))
#         x = self.relu(self.fc3(encoded))
        x = self.relu(self.fc2(encoded))
        x = x.view(-1, 64, 17, 17)
        x = self.unpool1(x, poolIdx3)
        x = self.relu(self.bn2(self.deconv1(x)))
        x = self.unpool2(x, poolIdx2)
        x = self.relu(self.bn1(self.deconv2(x)))
        x = self.unpool3(x, poolIdx1)
        decoded = self.relu(self.deconv3(x))
        return decoded, encoded