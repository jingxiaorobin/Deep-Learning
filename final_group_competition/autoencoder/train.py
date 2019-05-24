import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.autograd import Variable
from tensorboardX import SummaryWriter

from data import data_transforms, validation_data_transforms
from model import ConvolutionalAutoEncoder
import random


writer = SummaryWriter()

# need to modify
# def kl_divergence(p, p_hat):
#     '''
#         p - sparsity constraint
#         q - p_hat, the average of activations of hidden units
#     '''
#     # incase for p and p_hat goes to zero
#     s1 = torch.sum(p * torch.log(p / p_hat))
#     s2 = torch.sum((1 - p) * torch.log((1 - p) / (1 - p_hat)))
#     return s1 + s2

# sparse auto encoder with L1 loss
# def sparse_loss_l1(autoencoder, images):
#     loss = 0
#     values = images
#     for i in range(2):
#         fc_layer = list(autoencoder.encoder.children())[2 * i]
#         relu = list(autoencoder.encoder.children())[2 * i + 1]
#         values = relu(fc_layer(values))
#         loss += torch.mean(torch.abs(values))
#     for i in range(1):
#         fc_layer = list(autoencoder.decoder.children())[2 * i]
#         relu = list(autoencoder.decoder.children())[2 * i + 1]
#         values = relu(fc_layer(values))
#         loss += torch.mean(torch.abs(values))
#     return loss



def train(epoch_num, model, train_loader, log_interval):
    # set up the training mode
    model.train()
    losses = []

    # setup the loss metric and optimizer
    loss_metric = nn.MSELoss()
    func = nn.Softmax(dim=1)
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for batch_idx, data in enumerate(train_loader):
        images, _ = data
        images = Variable(images).cuda()
        # images = images.view(images.size(0), -1).cuda()
    
        optimizer.zero_grad()
        # need to modify
        decoded, encoded = model(images)
        # mean square loss
        loss = loss_metric(decoded, images)
        # l1_loss = sparse_loss(model, images)
        # p = torch.FloatTensor([args.sparsity_constraint for _ in range(encoded.size(1))]).unsqueeze(0)
        # p_hat = torch.sum(encoded, dim=0, keepdim=True) / encoded.size(0)
        # p = func(p).cuda()
        # p_hat = func(p_hat).cuda()

        # kl_loss = kl_divergence(p, p_hat)
        # loss = mse_loss + kl_loss * args.sparse_reg_param
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        # Print loss (uncomment lines below once implemented)
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch_num, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    # calculate the avg loss over an epoch
    avg_train_loss = sum(losses) / float(len(train_loader.dataset))
    writer.add_scalar('Avg Loss', avg_train_loss, epoch_num)
    print('\nAverage Train Loss: {:.4f} epoch {}'.format(avg_train_loss, epoch_num))

    # ------------------------------------------------------------------------------
    # for testing, we save the model and test on MNIST
    # filepath = 'models/test/sparse_test_' + str(epoch_num) + '.pth'
    # torch.save(model.state_dict(), filepath)


def validation(epoch_num, model, val_loader):
    model.eval()
    validation_loss = 0

    loss_metric = nn.MSELoss()
    # func = nn.Softmax(dim=1)


    with torch.no_grad():
        for batch_idx, data in enumerate(val_loader):
            images, _ = data
            # images = images.view(images.size(0), -1)
            images = Variable(images).cuda()
            
            # need to modify
            decoded, encoded = model(images)
            
            # compute the loss
            loss = loss_metric(decoded, images)
            # kl loss
            # p = torch.FloatTensor([args.sparsity_constraint for _ in range(encoded.size(1))]).unsqueeze(0)
            # p_hat = torch.sum(encoded, dim=0, keepdim=True) / encoded.size(0)
            # p = func(p).cuda()
            # p_hat = func(p_hat).cuda()
            # kl_loss = kl_divergence(p, p_hat)
            # l1_loss = sparse_loss(model, images)
            validation_loss += loss
            # save the reconstruction images (only for first 16 images)
            if batch_idx == 1:
                n = min(images.size(0), 16)
                comparison = torch.cat([images.view(args.batch_size, 3, 96, 96)[:n], decoded.view(args.batch_size, 3, 96, 96)[:n]])
                # need to modify
                save_image(comparison.cpu(),
                            'results/reconstruction1_covae_train_' + str(epoch_num) + '.png', nrow=n)

    avg_loss = validation_loss / float(len(val_loader.dataset)) # avg validation loss for all the validation set
    # writer.add_scalar('Val Loss cumulative', validation_loss, epoch_num) # cumulative
    writer.add_scalar('Avg val loss', avg_loss, epoch_num) # avg val loss over an epoch
    print('\nAverage Val Loss: {:.4f} epoch {}'.format(avg_loss, epoch_num))

    # global BEST_VAL
    
    # need to modify
    # if avg_loss < BEST_VAL:
    # BEST_VAL = avg_loss
    filepath = 'models/train/covae_model_train_' + str(epoch_num) + '.pth'
    torch.save(model.state_dict(), filepath)
    print('Save Best Model in HISTORY for epoch num {}\n'.format(epoch_num))


    
def run(args):
    # load the training set and validation set of unlabelled data
    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder('../../'+args.data+'/', transform=data_transforms),
        batch_size=args.batch_size, shuffle=True, num_workers=8)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder('../../supervised/val/', transform=validation_data_transforms),
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


    fc_input_features = 17*17*64 # input dimension for 96 * 96 images
    model = nn.DataParallel(ConvolutionalAutoEncoder(fc_input_features))
    model.cuda()

    for epoch in range(1, args.epochs + 1):
        train(epoch, model, train_loader, args.log_interval)
        validation(epoch, model, val_loader)    
        
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="sparse-autoencoder")
    parser.add_argument('--data', type=str, default='unsupervised', metavar='D',
                        help="folder where data is located. train_data.zip and test_data.zip need to be found in the folder")
    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 10)')
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
