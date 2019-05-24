import torch
import torch.nn as nn


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
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

        self.inv_fc = nn.Linear(1024, 2048)
        self.inv_fc1 = nn.Linear(2048, 32 * 42 * 42)
        self.relu5 = nn.ReLU()

        self.deconv1 = nn.ConvTranspose2d(32, 16, kernel_size=3)
        self.bn4 = nn.BatchNorm2d(16)
        self.relu6 = nn.ReLU()

        self.deconv2 = nn.ConvTranspose2d(16, 8, kernel_size=3)
        self.bn5 = nn.BatchNorm2d(8)
        self.relu7 = nn.ReLU()

        self.unpool1 = nn.MaxUnpool2d(2)
        self.deconv3 = nn.ConvTranspose2d(8, 3, kernel_size=5)

    def encode(self, x):
        h1, indices = self.pool1(self.bn1(self.conv1(x)))
        h1 = self.relu1(h1)

        h1 = self.relu2(self.bn2(self.conv2(h1)))
        h1 = self.relu3(self.bn3(self.conv3(h1)))

        h1 = h1.view(-1, 32 * 42 * 42)
        h1 = self.relu3(self.fc(h1))
        return self.fc21(h1), self.fc22(h1), indices

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, indices):
        h3 = self.inv_fc1(self.relu4(self.inv_fc(z)))
        h3 = h3.view(-1, 32, 42, 42)
        h3 = self.relu6(self.bn4(self.deconv1(h3)))
        h3 = self.relu7(self.bn5(self.deconv2(h3)))
        return torch.sigmoid(self.deconv3(self.unpool1(h3, indices)))

    def forward(self, x):
        mu, logvar, indices = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, indices), mu, logvar
