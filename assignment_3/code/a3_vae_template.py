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
        self.lin_log_std = nn.Linear(hidden_dim, z_dim)
        self.lin_mean = nn.Linear(hidden_dim, z_dim)

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



    def forward(self, input):
        """
        Given input, perform an encoding and decoding step and return the
        negative average elbo for the given batch.
        """
        negative_elbos = torch.zeros(1)
        for img in input:
            img = img.view(784)
            mean, std = self.encoder(img)
            sigma = torch.diag(std)
            l_reg = 0.5*(1+torch.log(std.pow(2))-mean.pow(2) - std.pow(2)).sum() 
            rand = torch.from_numpy(np.random.normal(0, 1, size=self.z_dim)).float()
            epsilons = torch.autograd.Variable(rand, requires_grad=False)
            z = mean + sigma * epsilons
            y = self.decoder(z)
            l_recon = (-img * torch.log(y) + (1-img) * torch.log(1-y)).mean()
            negative_elbos += -(l_reg + l_recon)
        average_negative_elbo = negative_elbos/len(input)
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
    epoch_elbos = torch.zeros(1)
    for batch in data:
        model.zero_grad()
        epoch_elbo = model(batch)
        print(epoch_elbo)
        optimizer.step()
        epoch_elbo.backward()
        epoch_elbos += epoch_elbo
    average_epoch_elbo = epoch_elbo/len(data)
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
        print("[Epoch {}] train elbo: {} val_elbo: {}".format(epoch, train_elbo, val_elbo))

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
