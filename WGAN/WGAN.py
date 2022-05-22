import os
import numpy as np
import pandas as pd
import math
import sys
from time import time
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

os.makedirs("images", exist_ok=True)

class Opt(object):
    dim = 10
    n_epochs = 200 #200
    batch_size = dim*dim
    lr = 0.00002
    n_cpu = 1
    latent_dim = 2 #100
    img_size = 28
    channels = 1
    n_critic = 5
    clip_value = 0.01
    sample_interval = 400
opt = Opt()

img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = True if torch.cuda.is_available() else False

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        self.model = nn.Sequential(
            *block(opt.latent_dim, 32, normalize=False),
            *block(32, 64),
            *block(64, 128),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )
    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *img_shape)
        return img

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(opt.img_size ** 2, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )
    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        validity = self.model(img_flat)
        return validity

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()

# Configure data loader
bs = 100

# MNIST Dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5), std=(0.5))])

train_dataset = datasets.MNIST(root='./mnist_data/', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./mnist_data/', train=False, transform=transform, download=False)

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=bs, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=bs, shuffle=False)

# Optimizers
generator_optimizer = torch.optim.RMSprop(generator.parameters(), lr=opt.lr)
discriminator_optimizer = torch.optim.RMSprop(discriminator.parameters(), lr=opt.lr)

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ----------
#  Training
# ----------

D = []
G = []

D_test = []
G_test = []

batches_done = 0
saved_imgs = []
for epoch in range(opt.n_epochs):
    print('Epoch ' + str(epoch) + ' training...' , end=' ')
    start = time()
    D_losses, G_losses = [], []
    D_losses_test, G_losses_test = [], []

    for i, (imgs, _) in enumerate(train_loader):
        real_imgs = Variable(imgs.type(Tensor))
        # train Discriminator
        discriminator_optimizer.zero_grad()
        # sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))
        # generate a batch of images
        fake_imgs = generator(z).detach()
        # Adversarial loss
        discriminator_loss = torch.mean(discriminator(fake_imgs)) - torch.mean(discriminator(real_imgs))

        D_losses.append(discriminator_loss)

        discriminator_loss.backward()
        discriminator_optimizer.step()
        # clip weights of discriminator
        for p in discriminator.parameters():
            p.data.clamp_(-opt.clip_value, opt.clip_value)
        # train the generator every n_critic iterations
        if i % opt.n_critic == 0:
            # train Generator
            generator_optimizer.zero_grad()
            # generate a batch of fake images
            critics_fake_imgs = generator(z)
            # Adversarial loss
            generator_loss = -torch.mean(discriminator(critics_fake_imgs))

            G_losses.append(generator_loss)

            generator_loss.backward()
            generator_optimizer.step()
        batches_done += 1
    end = time()
    elapsed = end - start

    for i, (imgs_test, _) in enumerate(test_loader):
        real_imgs_test = Variable(imgs_test.type(Tensor))
        # sample noise as generator input
        z_test = Variable(Tensor(np.random.normal(0, 1, (imgs_test.shape[0], opt.latent_dim))))
        # generate a batch of images
        fake_imgs_test = generator(z_test).detach()
        # Adversarial loss
        discriminator_loss_test = torch.mean(discriminator(fake_imgs_test)) - torch.mean(discriminator(real_imgs_test))
        critics_fake_imgs_test = generator(z_test)
        # Adversarial loss
        generator_loss_test = -torch.mean(discriminator(critics_fake_imgs_test))

        G_losses_test.append(generator_loss_test)

        D_losses_test.append(discriminator_loss_test)






    D_losses_mean = torch.mean(torch.FloatTensor(D_losses))
    G_losses_mean = torch.mean(torch.FloatTensor(G_losses))

    D_losses_mean_test = torch.mean(torch.FloatTensor(D_losses_test))
    G_losses_mean_test = torch.mean(torch.FloatTensor(G_losses_test))

    D.append(D_losses_mean.item())
    G.append(G_losses_mean.item())

    D_test.append(D_losses_mean_test.item())
    G_test.append(G_losses_mean_test.item())


    print('done, took %.1f seconds.' % elapsed)
    print('[%d/%d]: loss_d: %.3f, loss_g: %.3f' % (
            (epoch), opt.n_epochs, D_losses_mean, G_losses_mean))

    grid = torchvision.utils.make_grid(critics_fake_imgs.data.cpu(), nrow=opt.dim)
    img = (np.transpose(grid.detach().numpy(), (1, 2 ,0)) * 255).astype(np.uint8)
    saved_imgs.append(img)





torch.save(discriminator, "W_discriminator_complex")
torch.save(generator, "W_generator_complex")





losses = []

losses.append(D)
losses.append(G)


losses.append(D_test)
losses.append(G_test)

numpy_array = np.array(losses)
transpose = numpy_array.T

transpose_list = transpose.tolist()

df = pd.DataFrame(transpose_list, columns = ["train_D", "train_G", "test_D", "test_G"])
df["epochs"] = range(0,opt.n_epochs)

df.plot(x="epochs", y= ["train_D", "train_G", "test_D", "test_G"], kind="line")
plt.show()
