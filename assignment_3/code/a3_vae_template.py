import argparse

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

from datasets.bmnist import bmnist
import numpy as np
class Encoder(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20):
        super().__init__()

        self.relu = nn.ReLU()
        self.linear = nn.Linear(784, hidden_dim)
        self.lin_log_std = nn.Linear(hidden_dim, z_dim*2)
        self.lin_mean = nn.Linear(hidden_dim, z_dim*2)

    def forward(self, input):
        """
        Perform forward pass of encoder.

        Returns mean and std with shape [batch_size, z_dim]. Make sure
        that any constraints are enforced.
        """
        out = self.relu(self.linear(input))
        mean = self.lin_mean(out)
        std = self.lin_log_std(out)
        return mean, std.exp()


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


    def kl_divergence(sigma_1, mu_1, sigma_2, mu_2):
        return torch.log(np.log(sigma_2)/np.log(sigma_1))+ (sigma_1+(mu_1-mu_2).pow(2))/(2*sigma_2) - (1/2)

    def forward(self, input):
        """
        Given input, perform an encoding and decoding step and return the
        negative average elbo for the given batch.
        """

        mean, std = self.encoder(input)
        sigma = torch.diag(std)
        univariate_mean = torch.zeros(self.z_dim)
        univariate_sigma = torch.diag(torch.ones(self.z_dim))

        l_reg = kl_divergence(sigma, mean, univariate_sigma , univariate_mean)
        rand = torch.from_numpy(np.random.normal(0, 1, size=self.z_dim)).float()
        epsilons = torch.autograd.Variable(rand, requires_grad=False)
        z = mean_pred + sigma * epsilons
        mean = self.decoder(z)
        l_recon = -input * torch.log(mean) + (1-input) * torch.log(1-mean)
        average_negative_elbo = (l_reg + l_recon).mean()
        return average_negative_elbo

    def sample(self, n_samples):
        """
        Sample n_samples from the model. Return both the sampled images
        (from bernoulli) and the means for these bernoullis (as these are
        used to plot the data manifold).
        """
        sampled_ims, im_means = None, None
        raise NotImplementedError()

        return sampled_ims, im_means


def epoch_iter(model, data, optimizer):
    """
    Perform a single epoch for either the training or validation.
    use model.training to determine if in 'training mode' or not.

    Returns the average elbo for the complete epoch.
    """
    epoch_elbos = []
    for batch in data:
        model.zero_grad()
        epoch_elbo = model(data)
        optimizer.step()
        average_epoch_elbo.backward()
        elbo_batch.append(epoch_elbo)
    average_epoch_elbo = epoch_elbo.mean()
    return average_epoch_elbo


def run_epoch(model, data, optimizer):
    """
    Run a train and validation epoch and return average elbo for each.
    """
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

    train_curve, val_curve = [], []
    for epoch in range(ARGS.epochs):
        elbos = run_epoch(model, data, optimizer)
        train_elbo, val_elbo = elbos
        train_curve.append(train_elbo)
        val_curve.append(val_elbo)
        print(f"[Epoch {epoch}] train elbo: {train_elbo} val_elbo: {val_elbo}")

        # --------------------------------------------------------------------
        #  Add functionality to plot samples from model during training.
        #  You can use the make_grid functioanlity that is already imported.
        # --------------------------------------------------------------------

    # --------------------------------------------------------------------
    #  Add functionality to plot plot the learned data manifold after
    #  if required (i.e., if zdim == 2). You can use the make_grid
    #  functionality that is already imported.
    # --------------------------------------------------------------------

    save_elbo_plot(train_curve, val_curve, 'elbo.pdf')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=40, type=int,
                        help='max number of epochs')
    parser.add_argument('--zdim', default=20, type=int,
                        help='dimensionality of latent space')

    ARGS = parser.parse_args()

    main()
