import argparse

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from torchvision.utils import save_image

from datasets.bmnist import bmnist
import numpy as np
class Encoder(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20):
        super().__init__()

        self.relu = nn.ReLU()
        self.linear = nn.Linear(784, hidden_dim)
        self.lin_std = nn.Linear(hidden_dim, z_dim)
        self.lin_mean = nn.Linear(hidden_dim, z_dim)

    def forward(self, input):
        """
        Perform forward pass of encoder.

        Returns mean and std with shape [batch_size, z_dim]. Make sure
        that any constraints are enforced.
        """
        out = self.relu(self.linear(input))
        mean = self.lin_mean(out)
        std = self.lin_std(out)
        return mean, std


class Decoder(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20):
        super().__init__()

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Linear(z_dim, hidden_dim)
        self.lin_mean = nn.Linear(hidden_dim, 784)

    def forward(self, input):
        """
        Perform forward pass of encoder.

        Returns mean with shape [batch_size, 784].
        """
        out = self.relu(self.linear(input))
        mean = self.sigmoid(self.lin_mean(out))
        return mean


class VAE(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20):
        super().__init__()

        self.z_dim = z_dim
        self.encoder = Encoder(hidden_dim, z_dim)
        self.decoder = Decoder(hidden_dim, z_dim)



    def forward(self, input):
        """
        Given input, perform an encoding and decoding step and return the
        negative average elbo for the given batch.
        """
        input = input.view(-1,784)
        mean, std = self.encoder(input)
        l_reg = 0.5*(1+torch.log(1e-8+std.pow(2))-mean.pow(2) - std.pow(2)).sum(1)
        rand = torch.from_numpy(np.random.normal(0, 1, size=(input.shape[0],self.z_dim))).float()
        epsilons = torch.autograd.Variable(rand, requires_grad=False)
        z = mean + std * epsilons
        y = self.decoder(z)
        l_recon = (input * torch.log(y) + (1-input) * torch.log(1-y)).sum(1)
        average_negative_elbo = -(l_reg.mean() + l_recon.mean())
        return average_negative_elbo

    def sample(self, n_samples):
        """
        Sample n_samples from the model. Return both the sampled images
        (from bernoulli) and the means for these bernoullis (as these are
        used to plot the data manifold).
        """
        rand = torch.from_numpy(np.random.normal(0, 1, size=(n_samples,self.z_dim))).float()
        im_means = torch.autograd.Variable(rand, requires_grad=False)
        sampled_ims = self.decoder(im_means)
        return sampled_ims, im_means


def epoch_iter(model, data, optimizer):
    """
    Perform a single epoch for either the training or validation.
    use model.training to determine if in 'training mode' or not.

    Returns the average elbo for the complete epoch.
    """
    epoch_elbos = torch.zeros(1)
    for batch in data:
        optimizer.zero_grad()
        epoch_elbo = model(batch)
        epoch_elbo.backward()
        optimizer.step()
        epoch_elbos += epoch_elbo.item()
    average_epoch_elbo = epoch_elbo/len(data)
    return average_epoch_elbo


def run_epoch(model, data, optimizer):
    """
    Run a train and validation epoch and return average elbo for each.
    """
    train_elbo, val_elbo = None, None
    traindata, valdata = data

    model.train()
    train_elbo = epoch_iter(model, traindata, optimizer)

    model.eval()
    val_elbo = epoch_iter(model, valdata, optimizer)

    return train_elbo, val_elbo


def save_elbo_plot(train_curve, val_curve, filename):
    plt.figure(figsize=(12, 6))
    plt.plot(train_curve, label='train elbo')
    plt.plot(val_curve, label='validation elbo')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('ELBO')
    plt.tight_layout()
    plt.savefig(filename)


def main():
    data = bmnist()[:2]  # ignore test split
    model = VAE(z_dim=ARGS.zdim)
    optimizer = torch.optim.Adam(model.parameters())

    n_samples = 5
    train_curve, val_curve = [], []

    for epoch in range(ARGS.epochs):
        sampled_ims, im_means =  model.sample(n_samples)
        save_image(make_grid(sampled_ims.view(n_samples,1,28,28)), ARGS.save_path+'sample_epoch_{}.png'.format(epoch), padding=4)
        elbos = run_epoch(model, data, optimizer)
        train_elbo, val_elbo = elbos
        train_curve.append(train_elbo)
        val_curve.append(val_elbo)
        print("[Epoch {}] train elbo: {} val_elbo: {}".format(epoch, train_elbo, val_elbo))
        
        # --------------------------------------------------------------------
        #  Add functionality to plot samples from model during training.
        #  You can use the make_grid functioanlity that is already imported.
        # --------------------------------------------------------------------
        
    #if zdim == 2:

    
    # --------------------------------------------------------------------
    #  Add functionality to plot plot the learned data manifold after
    #  if required (i.e., if zdim == 2). You can use the make_grid
    #  functionality that is already imported.
    # --------------------------------------------------------------------

    save_elbo_plot(train_curve, val_curve, ARGS.save_path+'elbo.pdf')

    torch.save_state_dict(ARGS.save_path+'model.pth')
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=40, type=int,
                        help='max number of epochs')
    parser.add_argument('--zdim', default=20, type=int,
                        help='dimensionality of latent space')
    parser.add_argument('--save_path', type=str, default='files/', help='path for saving files')
    

    ARGS = parser.parse_args()

    main()
