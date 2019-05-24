import torch
import torch.nn as nn


class VAEClassifier(nn.Module):
    def __init__(self):
        super(VAEClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(8)
        self.pool1 = nn.MaxPool2d(2, return_indices=True)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(8, 16, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(16)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(16, 32, kernel_size=3)
        self.bn3 = nn.BatchNorm2d(32)
        self.relu3 = nn.ReLU()

        self.fc = nn.Linear(32 * 42 * 42, 2048)
        self.relu4 = nn.ReLU()

        self.fc21 = nn.Linear(2048, 1024)
        self.fc22 = nn.Linear(2048, 1024)

    def initialize(self):
        self.out_fc1 = nn.Linear(1024, 8192)
        self.relu5 = nn.ReLU()
        self.out_fc2 = nn.Linear(8192, 2000)

    def encode(self, x):
        h1, indices = self.pool1(self.bn1(self.conv1(x)))
        h1 = self.relu1(h1)

        h1 = self.relu2(self.bn2(self.conv2(h1)))
        h1 = self.relu3(self.bn3(self.conv3(h1)))

        h1 = h1.view(-1, 32 * 42 * 42)
        h1 = self.relu4(self.fc(h1))
        return self.fc21(h1), self.fc22(h1), indices

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def classify(self, z):
        out = self.relu5(self.out_fc1(z))
        return self.out_fc2(out)

    def forward(self, x):
        mu, logvar, _ = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.classify(z)
