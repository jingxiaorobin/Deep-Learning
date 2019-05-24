import torch
import torch.nn as nn
from torchvision.models import resnet18

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.input_features = 96
        self.fc1 = nn.Linear(self.input_features * self.input_features, 4096)
        self.relu1 = nn.ReLU()

        self.fc21 = nn.Linear(4096, 2048)
        self.fc22 = nn.Linear(4096, 2048)

        self.fc = nn.Linear(2048, self.input_features * self.input_features)
        self.out_fc = nn.Linear(3 * self.input_features * self.input_features, 1000)

        # Load pre-trained model
        self.load_weights('variational-autoencoder/models_classify/model_None_8.pth')

    def load_weights(self, pretrained_model_path, cuda=True):
        # Load pretrained model
        pretrained_model = torch.load(f=pretrained_model_path, map_location="cuda" if cuda else "cpu")

        # Load pre-trained weights in current model
        with torch.no_grad():
            self.load_state_dict(pretrained_model, strict=True)

        # Debug loading
        print('Parameters found in pretrained model:')
        pretrained_layers = pretrained_model.keys()
        for l in pretrained_layers:
            print('\t' + l)
        print('')

        for name, module in self.state_dict().items():
            if name in pretrained_layers:
                assert torch.equal(pretrained_model[name].cpu(), module.cpu())
                print('{} have been loaded correctly in current model.'.format(name))
            else:
                raise ValueError("state_dict() keys do not match")

    def encode(self, x):
        h1 = self.relu1(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        batch_size = x.size(0)
        mu, logvar = self.encode(x.view(-1, self.input_features * self.input_features))
        z = self.reparameterize(mu, logvar)
        out = self.fc(z)
        out = out.view(batch_size, 3 * self.input_features * self.input_features)
        return self.out_fc(out)
