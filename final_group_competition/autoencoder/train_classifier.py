import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from tensorboardX import SummaryWriter

from data import data_transforms, validation_data_transforms
from model import ConvolutionalAutoEncoder
from cae_classifier import CAEClassifier
from utils import AverageMeter, accuracy


writer = SummaryWriter()


def train(epoch_num, model, train_loader, log_interval):
    # set up the training mode
    model.train()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # setup the loss metric and optimizer
    loss_metric = nn.NLLLoss()
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for batch_idx, data in enumerate(train_loader):
        images, targets = data
        images = Variable(images).cuda()
        targets = Variable(targets).cuda()
    
        optimizer.zero_grad()
        outputs = model(images)
        # NLL loss
        loss = loss_metric(outputs, targets)
        loss.backward()
        optimizer.step()
        
        # update the loss and compute the accuracy
        losses.update(loss.item(), images.size(0))
        acc1 = accuracy(outputs, targets, 1)
        acc5 = accuracy(outputs, targets, 5)
        top1.update(acc1, images.size(0))
        top5.update(acc5, images.size(0))

        # Print loss (uncomment lines below once implemented)
        print('Validation Epoch: {} [{}/{} ({:.0f}%)]\t'
          'Loss: {loss.val:.4f} ({loss.avg:.4f})\t'
          'Top 1 Accuracy {top1.val:.3f} ({top1.avg:.3f})\t'
          'Top 5 Accuracy {top5.val:.3f} ({top5.avg:.3f})'.format(
              epoch_num, batch_idx * len(data), len(train_loader.dataset),
              100. * batch_idx / len(train_loader),
              loss=losses, top1=top1, top5=top5))
    # Record the metrics
    writer.add_scalar('Training loss', losses.avg, epoch_num)
    writer.add_scalar('Training_top1_acc', top1.avg, epoch_num)
    writer.add_scalar('Training_top5_acc', top5.avg, epoch_num)

    # for testing, we save the model and test on MNIST
    # filepath = 'models/test/sparse_test_' + str(epoch_num) + '.pth'
    # torch.save(model.state_dict(), filepath)

def validate(epoch_num, model, val_loader):
    model.eval()

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    loss_metric = nn.NLLLoss()

    with torch.no_grad():
        for batch_idx, data in enumerate(val_loader):
            images, targets = data
            images = Variable(images).cuda()
            targets = Variable(targets).cuda()

            outputs = model(images)
            loss = loss_metric(outputs, targets)
            losses.update(loss.item(), images.size(0))
            acc1 = accuracy(outputs, targets, 1)
            acc5 = accuracy(outputs, targets, 5)
            top1.update(acc1, images.size(0))
            top5.update(acc5, images.size(0))

    print('Validation Epoch: {} [{}/{} ({:.0f}%)]\t'
          'Loss: {loss.val:.4f} ({loss.avg:.4f})\t'
          'Top 1 Accuracy {top1.val:.3f} ({top1.avg:.3f})\t'
          'Top 5 Accuracy {top5.val:.3f} ({top5.avg:.3f})'.format(
              epoch_num, batch_idx * len(data), len(val_loader.dataset),
              100. * batch_idx / len(val_loader),
              loss=losses, top1=top1, top5=top5))
    # Record the metrics
    writer.add_scalar('Validation loss', losses.avg, epoch_num)
    writer.add_scalar('Validation_top1_acc', top1.avg, epoch_num)
    writer.add_scalar('Validation_top5_acc', top5.avg, epoch_num)

    global BEST_VAL
    
    if top1.avg < BEST_VAL:
        BEST_VAL = top1.avg
        filepath = 'models/train/classifier_model_' + str(epoch_num) + '.pth'
        torch.save(model.state_dict(), filepath)
        print('Save Best Model in HISTORY for epoch num {}\n'.format(epoch_num))


def run(args):
    # load the training set and validation set of unlabelled data
    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder('../../'+args.data+'/train', transform=data_transforms),
        batch_size=args.batch_size, shuffle=True, num_workers=8)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder('../../'+args.data+'/val', transform=validation_data_transforms),
        batch_size=args.batch_size, shuffle=False, num_workers=8)

    # try it on MNIST to see if the sparse ae can work
    # Load the MNIST training set with batch size 128, apply data shuffling and normalization
    # train_loader = torch.utils.data.DataLoader(
    #     datasets.MNIST('../data', train=True, download=True,
    #                 transform=transforms.Compose([
    #                     transforms.ToTensor(),
    #                     transforms.Normalize((0.1307,), (0.3081,))
    #                 ])),
    #     batch_size=128, shuffle=True)

    # val_loader = torch.utils.data.DataLoader(
    #     datasets.MNIST('../data', train=False, transform=transforms.Compose([
    #                     transforms.ToTensor(),
    #                     transforms.Normalize((0.1307,), (0.3081,))
    #                 ])),
    #     batch_size=128, shuffle=True)


    # input_dim = 96*96*3 # need to change
    fc_input_features = 17*17*64
    n_classes = 1000

    pretrained_dict = torch.load(args.pth, map_location=args.device)
    
    classifier = CAEClassifier(fc_input_features, n_classes)
    classifier = nn.DataParallel(classifier)
    model_dict = classifier.state_dict()

    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict) 
    classifier.load_state_dict(pretrained_dict)

    for param in classifier.module.parameters():
        param.requires_grad = False

    classifier.module.initialize()
    classifier = classifier.to(args.device)

    for epoch in range(1, args.epochs + 1):
        train(epoch, classifier, train_loader, args.log_interval)
        validate(epoch, classifier, val_loader)  
        
    writer.close()







if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="sparse-autoencoder-classifier")
    parser.add_argument('--data', type=str, default='supervised', metavar='D',
                        help="folder where data is located. train_data.zip and test_data.zip need to be found in the folder")
    parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--pth', type=str, required=True,
                        help='the file path for loading model parameters')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--weight_decay', type=float, default=1e-5, metavar='WD',
                        help="weight decay parameter lambda for the regularization")
    parser.add_argument('--sparse_reg_param', type=float, default=1e-3, metavar='SW',
                        help="sparsity weigth parameter for sparse AE")
    parser.add_argument('--sparsity_constraint', type=float, default=0.01, metavar="SP",
                        help="sparsity constraint parameter")
    parser.add_argument('--momentum', type=float, default=0.0, metavar='M',
                        help='SGD momentum (default: 0.0)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    
    args = parser.parse_args()

    cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if cuda else "cpu")
    args.device = device

    BEST_VAL = float('inf')
    
    run(args)