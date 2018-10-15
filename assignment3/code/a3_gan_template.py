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
        nn.Linear(latent_dim, 128),
        nn.LeakyReLU(0.2, True),
        nn.Linear(128, 256),
        nn.BatchNorm1d(256),
        nn.LeakyReLU(0.2, True),
        nn.Linear(256, 512),
        nn.BatchNorm1d(512),
        nn.LeakyReLU(0.2, True),
        nn.Linear(512, 1024),
        nn.BatchNorm1d(1024),
        nn.LeakyReLU(0.2, True),
        nn.Linear(1024, 784),
        nn.Tanh())

    def forward(self, z):
        return self.layers(z)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.layers = nn.Sequential(
        nn.Linear(784, 512),
        nn.LeakyReLU(0.2, True),
        nn.Linear(512, 256),
        nn.LeakyReLU(0.2, True),
        nn.Linear(256, 1),
        nn.Sigmoid())

    def forward(self, img):
        return self.layers(img)


def train(adversarial_loss, dataloader, discriminator, generator, optimizer_G, optimizer_D, latent_dim):
    for epoch in range(args.n_epochs):
        for i, (imgs, _) in enumerate(dataloader):

            imgs.cuda()
            imgs = imgs.view(imgs.shape[0], 784)
            # Train Generator
            # ---------------
            valid = torch.autograd.Variable(torch.zeros(imgs.shape[0],1).uniform_(0.9,1.1), requires_grad=False)

            optimizer_G.zero_grad()

            z = torch.autograd.Variable(torch.randn(imgs.shape[0], latent_dim))

            gen_imgs = generator(z)
            dis_vals = discriminator(gen_imgs)
            loss_gen = adversarial_loss(dis_vals, valid)


            loss_gen.backward()
            optimizer_G.step()

            # Train Discriminator
            # -------------------

            fake = torch.autograd.Variable(torch.zeros(imgs.shape[0],1).uniform_(0.0,0.2), requires_grad=False)

            optimizer_D.zero_grad()

            valid_vals = discriminator(imgs)
            real_loss = adversarial_loss(valid_vals, valid)
            real_loss.backward(retain_graph=True)
            optimizer_D.step()

            optimizer_D.zero_grad()

            fake_vals = discriminator(gen_imgs.detach())
            fake_loss = adversarial_loss(fake_vals, fake)
            fake_loss.backward()
            optimizer_D.step()

            # Save Images
            # -----------
            batches_done = epoch * len(dataloader) + i
            print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" % (epoch, args.n_epochs, i, len(dataloader),
                                                                                    loss_dis.item(), loss_gen.item()))
            if batches_done % args.save_interval == 0:
                save_image(gen_imgs[:25].view(25, 1, 28, 28),
                            'images/{}.png'.format(batches_done),
                            nrow=5, normalize=True)

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

    adversarial_loss = torch.nn.BCELoss()
    # Start training
    train(adversarial_loss, dataloader, discriminator, generator, optimizer_G, optimizer_D, args.latent_dim)

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
