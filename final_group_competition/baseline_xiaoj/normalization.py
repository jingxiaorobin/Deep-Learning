import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from tensorboardX import SummaryWriter

from data import data_transforms, validation_data_transforms
from model import CNNModel


def normal(args):
    loader = torch.utils.data.DataLoader(\
            datasets.ImageFolder('../../ssl_data_96/supervised/train', transform=data_transforms),\
            num_workers=4)                #n_worker to 4, to use 4 gpu


    mean = 0.
    std = 0.
    nb_samples = 0.
    for i in loader:
        for data in i:
            batch_samples = data.size(0)
            data = data.view(batch_samples, data.size(1), -1)
            mean += data.mean(2).sum(0)
            std += data.std(2).sum(0)
            nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples
    print(mean)
    print(std)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CNN Baseline')
    parser.add_argument('--data', type=str, default='data', metavar='D',
                        help="folder where data is located. train_data.zip and test_data.zip need to be found in the folder")
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=25, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.1, metavar='M',
                        help='SGD momentum (default: 0.1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
normal(args)