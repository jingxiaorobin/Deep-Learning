import argparse
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torchvision import datasets, transforms
from torchvision.utils import save_image

from model import SWWAE


writer = SummaryWriter()


def train(epoch, model, train_loader, unlabeled_loader, optimizer, args):
    model.train()
    crit = nn.MSELoss()

    for label, unlabeled in zip(enumerate(train_loader), enumerate(unlabeled_loader)):
        batch_idx, (data, target) = label
        n = data.size(0)
        batch_idx, (data_un, _) = unlabeled
        data = torch.cat((data, data_un), 0)
        target = torch.cat((target, torch.LongTensor(96).fill_(-1)), 0)

        data, target = data.to(args.device), target.to(args.device)
        optimizer.zero_grad()
        out_decode, output = model(data)

        loss_delta = crit(out_decode, data)
        loss = F.nll_loss(output[0:n], target[0:n]) + loss_delta
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * n, n*len(train_loader),
                100. * batch_idx / len(train_loader), loss.item()))


def test(epoch, model, loader, args):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for i, (data, target) in enumerate(loader):
            data, target = data.to(args.device), target.to(args.device)
            _, output = model(data)
            test_loss += F.nll_loss(output, target).item()
            pred = output.data.max(1)[1]
            correct += pred.eq(target.data).cpu().sum()
    test_loss /= len(loader)
    print('====> Accuracy: {}'.format(correct / (args.batch_size * len(loader))))
    print('====> Test set loss: {:.4f}'.format(test_loss))


def run(args):
    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(args.data, transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=True, num_workers=4)
    unlabeled_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(args.data_unlabeled, transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=True, num_workers=4)

    test_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(args.data_test, transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = SWWAE()
    model = model.to(args.device)

    if args.checkpoint is not None:
        model.load_state_dict(torch.load(args.checkpoint))

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    for epoch in range(1, args.epochs + 1):
        train(epoch, model, train_loader, unlabeled_loader, optimizer, args)
        model_file = 'models/model_' + str(epoch) + '.pth'
        torch.save(model.state_dict(), model_file)
        print('Saved model to ' + model_file)
        test(epoch, model, test_loader, args)
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stacked What Where Autoencoder")
    parser.add_argument('--data', type=str, required=True,
                        help="folder where labeled data is located")
    parser.add_argument('--data-unlabeled', type=str, required=True,
                        help="folder where unlabeled data is located")
    parser.add_argument('--data-test', type=str, required=True,
                        help="folder where test data is located")
    parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='path to model checkpoint')
    parser.add_argument('--lr', type=float, default=1e-2, metavar='LR',
                        help='learning rate (default: 1e-2)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.0)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--no-cuda', action="store_true",
                        help='run on cpu')

    args = parser.parse_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print('Running with these options:', args)
    run(args)
