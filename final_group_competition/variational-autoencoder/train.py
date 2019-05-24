import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets
from torchvision.transforms import transforms
from tensorboardX import SummaryWriter
from torchvision.utils import save_image
from tqdm import tqdm

from model import VAE


writer = SummaryWriter()


def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x.view(-1, 96*96), x.view(-1, 96*96), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


def train(epoch_num, model, train_loader, optimizer, args):
    model.train()
    losses = []
    for batch_idx, (images, _) in tqdm(enumerate(train_loader), total=len(train_loader)):
        images = images.to(args.device)

        optimizer.zero_grad()
        recon_batch, mu, logvar = model(images)
        loss = loss_function(recon_batch, images, mu, logvar)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{}]\tLoss: {:.6f}'.format(
                epoch_num, batch_idx, len(train_loader), loss.item()))
    writer.add_scalar('Avg Loss', sum(losses) / float(len(losses)))


def test(epoch, model, loader, args):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(loader):
            data = data.to(args.device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                      recon_batch.view(args.batch_size, 3, 96, 96)[:n]])
                save_image(comparison.cpu(),
                         'results/reconstruction_' + str(epoch) + '.png', nrow=n)
                break
    test_loss /= len(loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


def run(args):
    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(args.data, transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=True, num_workers=8)

    test_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(args.data_val, transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=False, num_workers=8)

    model = nn.DataParallel(VAE())
    model = model.to(args.device)

    if args.checkpoint is not None:
        model.load_state_dict(torch.load(args.checkpoint))

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    for epoch in range(1, args.epochs + 1):
        train(epoch, model, train_loader, optimizer, args)
        model_file = 'models/model_' + str(epoch) + '.pth'
        torch.save(model.state_dict(), model_file)
        print('Saved model to ' + model_file)
        test(epoch, model, test_loader, args)
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Variational Autoencoder")
    parser.add_argument('--data', type=str, required=True,
                        help="folder where data is located")
    parser.add_argument('--data-val', type=str, required=True,
                        help='folder where validation data is located')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='path to model checkpoint')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--no-cuda', action="store_true",
                        help='run on cpu')

    args = parser.parse_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print('Running with these options:', args)
    run(args)
