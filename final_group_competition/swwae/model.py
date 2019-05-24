import torch.nn as nn
import torch.nn.functional as F


class SWWAE(nn.Module):
    def __init__(self):
        super(SWWAE, self).__init__()
        # Convolution
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(4, return_indices=True)

        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(3, return_indices=True)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, return_indices=True)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.conv7 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm2d(256)
        self.conv8 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn8 = nn.BatchNorm2d(512)
        self.conv9 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn9 = nn.BatchNorm2d(512)
        self.conv10 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn10 = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d(2, return_indices=True)

        self.fc1 = nn.Linear(2048, 4096)
        self.fc2 = nn.Linear(4096, 1000)

        # Deconvolution
        self.fc3 = nn.Linear(1000, 4096)
        self.fc4 = nn.Linear(4096, 2048)

        self.unpool4 = nn.MaxUnpool2d(2)
        self.deconv10 = nn.ConvTranspose2d(512, 512, kernel_size=3, padding=1)
        self.bn11 = nn.BatchNorm2d(512)
        self.deconv9 = nn.ConvTranspose2d(512, 512, kernel_size=3, padding=1)
        self.bn12 = nn.BatchNorm2d(512)
        self.deconv8 = nn.ConvTranspose2d(512, 256, kernel_size=3, padding=1)
        self.bn13 = nn.BatchNorm2d(256)
        self.deconv7 = nn.ConvTranspose2d(256, 256, kernel_size=3, padding=1)
        self.bn14 = nn.BatchNorm2d(256)
        self.deconv6 = nn.ConvTranspose2d(256, 256, kernel_size=3, padding=1)
        self.bn15 = nn.BatchNorm2d(256)
        self.deconv5 = nn.ConvTranspose2d(256, 128, kernel_size=3, padding=1)
        self.bn16 = nn.BatchNorm2d(128)

        self.unpool3 = nn.MaxUnpool2d(2)
        self.deconv4 = nn.ConvTranspose2d(128, 128, kernel_size=3, padding=1)
        self.bn17 = nn.BatchNorm2d(128)
        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1)
        self.bn18 = nn.BatchNorm2d(64)

        self.unpool2 = nn.MaxUnpool2d(3)
        self.deconv2 = nn.ConvTranspose2d(64, 64, kernel_size=3, padding=1)
        self.bn19 = nn.BatchNorm2d(64)

        self.unpool1 = nn.MaxUnpool2d(4)
        self.deconv1 = nn.ConvTranspose2d(64, 3, kernel_size=3, padding=1)
        self.bn20 = nn.BatchNorm2d(3)

    def forward(self, x):
        x, indices1 = self.pool1(self.bn1(self.conv1(x)))
        x_conv1 = F.relu(x)

        x, indices2 = self.pool2(self.bn2(self.conv2(x_conv1)))
        x_conv2 = F.relu(x)

        x, indices3 = self.pool3(self.bn3(self.conv4(self.bn3(self.conv3(x_conv2)))))
        x_conv3 = F.relu(x)


        x = self.conv5(x_conv3)

        x = self.conv7(self.bn6(self.conv6(self.bn5(x))))

        x = self.conv9(self.bn8(self.conv8(self.bn7(x))))

        x, indices4 = self.pool4(self.bn10(self.conv10(self.bn9(x))))
        x_conv4 = F.relu(x)

        x = x_conv4.view(-1, 512*2*2)
        x = F.dropout(F.relu(self.fc1(x)), training=self.training)

        x_nll = self.fc2(x)

        x = self.fc3(x_nll)
        x_deconv4 = F.relu(self.fc4(x))
        x = x_deconv4.view(x_deconv4.size(0), 512, 2, 2)

        x = F.relu(self.bn16(self.deconv5(self.bn15(self.deconv6(self.bn14(self.deconv7(self.bn13(self.deconv8(self.bn12(self.deconv9(self.bn11(self.deconv10(self.unpool4(x, indices4))))))))))))))

        x = F.relu(self.bn18(self.deconv3(self.bn17(self.deconv4(self.unpool3(x, indices3))))))

        x = F.relu(self.bn19(self.deconv2(self.unpool2(x, indices2))))

        output = F.relu(self.bn20(self.deconv1(self.unpool1(x, indices1))))
        return output, F.log_softmax(x_nll)
