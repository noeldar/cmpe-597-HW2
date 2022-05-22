# prerequisites
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image
import os
import numpy as np
import pandas as pd
import math
import sys
import torchvision
from time import time
import matplotlib.pyplot as plt

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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


class Generator(nn.Module):
    def __init__(self, g_input_dim, g_output_dim):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(g_input_dim, 1024)
        self.fc2 = nn.Linear(self.fc1.out_features, g_output_dim)

    # forward method
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        return torch.tanh(self.fc2(x))


class Discriminator(nn.Module):
    def __init__(self, d_input_dim):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(d_input_dim, 512)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features // 2)
        self.fc3 = nn.Linear(self.fc2.out_features, 1)

    # forward method
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        return torch.sigmoid(self.fc3(x))


# build network
z_dim = 2
mnist_dim = train_dataset.train_data.size(1) * train_dataset.train_data.size(2)


print(train_dataset.train_data.size(1), train_dataset.train_data.size(2))

G = Generator(g_input_dim = z_dim, g_output_dim = mnist_dim).to(device)
D = Discriminator(mnist_dim).to(device)


# loss
criterion = nn.BCELoss()

# optimizer
lr = 0.0002
G_optimizer = optim.Adam(G.parameters(), lr = lr)
D_optimizer = optim.Adam(D.parameters(), lr = lr)


def D_train(x):
    # =======================Train the discriminator=======================#
    D.zero_grad()

    # train discriminator on real
    x_real, y_real = x.view(-1, mnist_dim), torch.ones(bs, 1)
    x_real, y_real = Variable(x_real.to(device)), Variable(y_real.to(device))

    D_output = D(x_real)
    D_real_loss = criterion(D_output, y_real)
    D_real_score = D_output

    # train discriminator on facke
    z = Variable(torch.randn(bs, z_dim).to(device))
    x_fake, y_fake = G(z), Variable(torch.zeros(bs, 1).to(device))

    D_output = D(x_fake)
    D_fake_loss = criterion(D_output, y_fake)
    D_fake_score = D_output

    # gradient backprop & optimize ONLY D's parameters
    D_loss = D_real_loss + D_fake_loss
    D_loss.backward()
    D_optimizer.step()

    return D_loss.data.item()


def G_train(x):
    # =======================Train the generator=======================#
    G.zero_grad()

    z = Variable(torch.randn(bs, z_dim).to(device))
    y = Variable(torch.ones(bs, 1).to(device))

    G_output = G(z)
    D_output = D(G_output)
    G_loss = criterion(D_output, y)

    # gradient backprop & optimize ONLY G's parameters
    G_loss.backward()
    G_optimizer.step()

    return [G_loss.data.item(), G_output]


def D_test(x):
    # train discriminator on real
    x_real, y_real = x.view(-1, mnist_dim), torch.ones(bs, 1)
    x_real, y_real = Variable(x_real.to(device)), Variable(y_real.to(device))

    D_output = D(x_real)
    D_real_loss = criterion(D_output, y_real)
    D_real_score = D_output

    # train discriminator on facke
    z = Variable(torch.randn(bs, z_dim).to(device))
    x_fake, y_fake = G(z), Variable(torch.zeros(bs, 1).to(device))

    D_output = D(x_fake)
    D_fake_loss = criterion(D_output, y_fake)
    D_fake_score = D_output

    # gradient backprop & optimize ONLY D's parameters
    D_loss = D_real_loss + D_fake_loss

    return D_loss.data.item()


def G_test(x):

    z = Variable(torch.randn(bs, z_dim).to(device))
    y = Variable(torch.ones(bs, 1).to(device))

    G_output = G(z)
    D_output = D(G_output)
    G_loss = criterion(D_output, y)

    return [G_loss.data.item(), G_output]


n_epoch = 50
batches_done = 0
saved_imgs = []
D_loss = []
G_loss = []

D_loss_test = []
G_loss_test = []

for epoch in range(1, n_epoch + 1):
    D_losses, G_losses = [], []
    D_losses_test, G_losses_test = [], []

    for batch_idx, (x, _) in enumerate(train_loader):
        D_losses.append(D_train(x))
        loss, out = G_train(x)
        G_losses.append(loss)

    for idx, (x_test, _) in enumerate(test_loader):
        D_losses_test.append(D_test(x_test))
        loss_test, out_test = G_test(x_test)
        G_losses_test.append(loss_test)

    print('[%d/%d]: loss_d: %.3f, loss_g: %.3f' % (
        (epoch), n_epoch, torch.mean(torch.FloatTensor(D_losses)), torch.mean(torch.FloatTensor(G_losses))))
    D_loss.append(torch.mean(torch.FloatTensor(D_losses)).item())
    G_loss.append(torch.mean(torch.FloatTensor(G_losses)).item())

    D_loss_test.append(torch.mean(torch.FloatTensor(D_losses_test)).item())
    G_loss_test.append(torch.mean(torch.FloatTensor(G_losses_test)).item())

    if epoch in [0, 4, 9, 49, 99, 149, 199]:
        grid = torchvision.utils.make_grid(out.data.cpu(), nrow=10)
        img = (np.transpose(grid.detach().numpy(), (1, 2, 0)) * 255).astype(np.uint8)
        saved_imgs.append(img)


torch.save(D, "discriminator_weak")
torch.save(G, "generator_weak")


losses = []

losses.append(D_loss)
losses.append(G_loss)


losses.append(D_loss_test)
losses.append(G_loss_test)

numpy_array = np.array(losses)
transpose = numpy_array.T

transpose_list = transpose.tolist()

df = pd.DataFrame(transpose_list, columns = ["train_D", "train_G", "test_D", "test_G"])
df["epochs"] = range(1, n_epoch + 1)

df.plot(x="epochs", y= ["train_D", "train_G", "test_D", "test_G"], kind="line")
plt.savefig('GAN_weak.png')
plt.show()
