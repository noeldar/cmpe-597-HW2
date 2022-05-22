import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch
import numpy as np
import torchvision
from GAN_Generator_complex import Generator
import matplotlib.pyplot as plt

from torchvision.models.inception import inception_v3

import numpy as np
from scipy.stats import entropy

def inception_score(imgs, cuda=True, batch_size=32, resize=False, splits=1):
    """Computes the inception score of the generated images imgs
    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    N = len(imgs)

    assert batch_size > 0
    assert N > batch_size

    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
    inception_model.eval();
    up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)
    conv = nn.Conv2d(1, 3, 1).type(dtype)

    def get_pred(x):
        x = up(x)
        x = conv(x)
        x = inception_model(x)
        return F.softmax(x).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))

    for i, batch in enumerate(dataloader, 0):
        batch = batch.type(dtype)
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]

        preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batchv)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)


def plot_random_latent_images(generator, name):
    for i in range(100):

        data = np.load(f"random_{i}.npy")

        if i == 0:
            z = data
        else:
            z = np.concatenate((z, data), axis=0)

    t = torch.from_numpy(z.astype(np.float32)).to("cuda")

    # generate a batch of images

    fake_imgs = generator(t).detach()
    fake_imgs = fake_imgs.view(fake_imgs.shape[0], 1, 28, 28)
    print(fake_imgs.shape)

    grid = torchvision.utils.make_grid(fake_imgs.data.cpu(), nrow=10)
    img = (np.transpose(grid.detach().numpy(), (1, 2, 0)) * 255).astype(np.uint8)

    plt.figure(figsize=(10, 10))

    plt.imshow(img, interpolation='nearest')

    plt.savefig(f'GAN_generatedImages_{name}.png')
    return fake_imgs

dim = 10
z_dim = 2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


G = Generator(g_input_dim = z_dim, g_output_dim = 28*28).to(device)
G = torch.load(f"generator_complex", map_location=torch.device('cuda'))
fake_imgs = plot_random_latent_images(G, "complex")

print("Calculating Inception Score...")
print(inception_score(fake_imgs, cuda=False, batch_size=32, resize=True, splits=10))

#(1.0000000000000053, 7.937900528762522e-15)
