import argparse
import os

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision import datasets
import numpy as np


class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.layers = nn.Sequential(
        nn.Linear(latent_dim, 128)
        nn.LeakyReLU(0.2, True),
        nn.Linear(128, 256)
        nn.BatchNorm2d(256),
        nn.LeakyReLU(0.2, True),
        nn.Linear(256, 512)
        nn.BatchNorm2d(512),
        nn.LeakyReLU(0.2, True),
        nn.Linear(512, 1024)
        nn.BatchNorm2d(1024),
        nn.LeakyReLU(0.2, True),
        nn.Linear(1024, 784),
        nn.ReLU(True))

    def forward(self, z):
        return self.layers(z)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.layers = nn.Sequential(
        nn.Linear(784, 512)
        nn.LeakyReLU(0.2, True),
        nn.Linear(512, 256)
        nn.LeakyReLU(0.2, True),
        nn.Linear(256, 1),
        nn.ReLU(True))

    def forward(self, img):
        return self.layers(z)


def train(dataloader, discriminator, generator, optimizer_G, optimizer_D, latent_dim, batch_size):
    for epoch in range(args.n_epochs):
        average_v = []
        for i, (imgs, _) in enumerate(dataloader):

            imgs.cuda()

            # Train Generator
            # ---------------
            optimizer_G.zero_grad()
            z = torch.randn_like(batch_size, latent_dim)
            gen_imgs = generator(z)
            loss_gen = -(imgs * torch.log(1e-8+y) + (1-imgs) * torch.log(1-y)).sum(1)
            loss_gen.backward()
            optimizer_G.step()

            # Train Discriminator
            # -------------------
            optimizer_D.zero_grad()
            gen_vals = discriminator(gen_imgs)
            loss_dis = torch.log(1- gen_vals)
            loss_dis.backward()
            optimizer_D.step()
            average_v.append(loss_gen+loss_dis)
            # Save Images
            # -----------
            batches_done = epoch * len(dataloader) + i
            if batches_done % args.save_interval == 0:
                save_image(gen_imgs[:25].view(batch_size, 1, 28, 28),
                            'images/{}.png'.format(batches_done),
                            nrow=5, normalize=True)

        print('Average min max value in epoch {}: {}'.format(epoch, np.average(average_v)))
def main(args):
    # Create output image directory
    os.makedirs('images', exist_ok=True)

    # load data
    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST('./data/mnist', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))])),
        batch_size=args.batch_size, shuffle=True)

    # Initialize models and optimizers
    generator = Generator(args.latent_dim)
    discriminator = Discriminator()
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr)

    # Start training
    train(dataloader, discriminator, generator, optimizer_G, optimizer_D, args.latent_dim, args.batch_size)

    # You can save your generator here to re-use it to generate images for your
    # report, e.g.:
    torch.save(generator.state_dict(), "mnist_generator.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=200,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='learning rate')
    parser.add_argument('--latent_dim', type=int, default=100,
                        help='dimensionality of the latent space')
    parser.add_argument('--save_interval', type=int, default=500,
                        help='save every SAVE_INTERVAL iterations')
    args = parser.parse_args()

    main(args)
