import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torchvision.transforms import transforms
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from tqdm import tqdm

from classifier import VAEClassifier
from utils import AverageMeter, accuracy


writer = SummaryWriter()


def train(epoch, model, loss_func, optimizer, data_loader, args):
    model.train()

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    for batch_idx, (data, target) in tqdm(enumerate(data_loader), total=len(data_loader)):
        data, target = Variable(data).cuda(), Variable(target).cuda()
        optimizer.zero_grad()
        output = model(data)

        loss = loss_func(output, target)
        loss.backward()
        optimizer.step()
        losses.update(loss.item(), data.size(0))
        acc1 = accuracy(output, target, 1)
        acc5 = accuracy(output, target, 5)
        top1.update(acc1, data.size(0))
        top5.update(acc5, data.size(0))

        if batch_idx % args.log_interval == 0:
            print('Train Batch: [{0}/{1}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top 1 Accuracy {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Top 5 Accuracy {top5.val:.3f} ({top5.avg:.3f})'.format(
                      batch_idx, len(data_loader), loss=losses, top1=top1, top5=top5))
    writer.add_scalar('train_loss', losses.avg, epoch)
    writer.add_scalar('train_top1_acc', top1.avg, epoch)
    writer.add_scalar('train_top5_acc', top5.avg, epoch)


def validate(epoch, model, loss_func, data_loader, args):
    model.eval()

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = Variable(data).cuda(), Variable(target).cuda()
            output = model(data)
            loss = loss_func(output, target)
            losses.update(loss.item(), data.size(0))
            acc1 = accuracy(output, target, 1)
            acc5 = accuracy(output, target, 5)
            top1.update(acc1, data.size(0))
            top5.update(acc5, data.size(0))
    print('Validation Epoch: {}\t'
          'Loss: {loss.avg:.4f}\t'
          'Top 1 Accuracy: {top1.avg:.3f}\t'
          'Top 5 Accuracy: {top5.avg:.3f}'.format(epoch, loss=losses, top1=top1, top5=top5))
    writer.add_scalar('val_loss', losses.avg, epoch)
    writer.add_scalar('val_top1_acc', top1.avg, epoch)
    writer.add_scalar('val_top5_acc', top5.avg, epoch)


def run(args):
    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(args.data + '/train', transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=True, num_workers=8)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(args.data + '/val', transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=True, num_workers=8)

    classifier = VAEClassifier()
    classifier = nn.DataParallel(classifier)

    pretrained_dict = torch.load(args.model)
    pretrained_dict = { k : v for k, v in pretrained_dict.items() if k in classifier.state_dict() }
    classifier.load_state_dict(pretrained_dict)
    for param in classifier.module.parameters():
        param.requires_grad = False
    classifier.module.initialize()

    classifier = classifier.to(args.device)

    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(classifier.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, args.step_size)

    for epoch in range(1, args.epochs + 1):
        scheduler.step()
        train(epoch, classifier, loss_func, optimizer, train_loader, args)
        acc = validate(epoch, classifier, loss_func, val_loader, args)
        model_file = 'models_classify/model_' + str(acc) + '_' + str(epoch) + '.pth'
        torch.save(classifier.state_dict(), model_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Variational Autoencoder for classification")
    parser.add_argument('--data', type=str, required=True,
                        help="folder where data is located")
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--model', type=str, required=True,
                        help='path to VAE model parameters')
    parser.add_argument('--step-size', type=int, default=5,
                        help='step size for learning rate annealing (default: 5)')
    parser.add_argument('--no-cuda', action="store_true",
                        help='run on cpu')

    args = parser.parse_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print('Running with these options:', args)
    run(args)
