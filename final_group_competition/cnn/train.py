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
from utils import AverageMeter, accuracy


writer = SummaryWriter()


def train(epoch, model, optimizer, train_loader, log_interval):
    model.train()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data).cuda(), Variable(target).cuda()
        optimizer.zero_grad()
        output = model(data)

        loss = F.cross_entropy(output, target, reduction='sum')
        loss.backward()
        optimizer.step()

        acc1 = accuracy(output, target, 1)
        acc5 = accuracy(output, target, 5)
        losses.update(loss.item(), data.size(0))
        top1.update(acc1, data.size(0))
        top5.update(acc5, data.size(0))
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    writer.add_scalar('Avg Loss', losses.avg, epoch)
    writer.add_scalar('Top 1 Accuracy', top1.avg, epoch)
    writer.add_scalar('Top 5 Accuracy', top5.avg, epoch)


def validation(epoch, model, val_loader):
    model.eval()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    with torch.no_grad():
        for data, target in val_loader:
            data, target = Variable(data).cuda(), Variable(target).cuda()
            output = model(data)
            loss = F.cross_entropy(output, target, reduction='sum')

            acc1 = accuracy(output, target, 1)
            acc5 = accuracy(output, target, 5)
            losses.update(loss.item(), data.size(0))
            top1.update(acc1, data.size(0))
            top5.update(acc5, data.size(0))

    writer.add_scalar('Val Loss', losses.avg, epoch)
    writer.add_scalar('Val Top 1 Accuracy', top1.avg, epoch)
    writer.add_scalar('Val Top 5 Accuracy', top5.avg, epoch)
    print('\nValidation set: Average loss: {:.4f}, Top 1 Accuracy: ({:.0f}%), Top 5 Accuracy: ({:.0f}%)\n'.format(
        losses.avg, top1.avg, top5.avg))


def run(args):
    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(args.data + '/train', transform=data_transforms),
        batch_size=args.batch_size, shuffle=True, num_workers=16)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(args.data + '/val', transform=validation_data_transforms),
        batch_size=args.batch_size, shuffle=False, num_workers=16)

    model = CNNModel()
    model = nn.DataParallel(model)
    model = model.to(args.device)

    if args.checkpoint is not None:
        model.load_state_dict(torch.load(args.checkpoint))

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size)

    for epoch in range(1, args.epochs + 1):
        scheduler.step()
        train(epoch, model, optimizer, train_loader, args.log_interval)
        validation(epoch, model, val_loader)
        model_file = 'model_' + str(epoch) + '.pth'
        torch.save(model.state_dict(), model_file)
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CNN Baseline')
    parser.add_argument('--data', type=str, default='data', metavar='D',
                        help="folder where data is located. train_data.zip and test_data.zip need to be found in the folder")
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--step-size', type=int, default=10)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=25, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=1e-2, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Running with these options:', args)
    run(args)
