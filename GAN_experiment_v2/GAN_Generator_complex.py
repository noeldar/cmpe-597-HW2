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


class Generator(nn.Module):
    def __init__(self, g_input_dim, g_output_dim):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(g_input_dim, 32)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features * 2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features * 2)
        self.fc4 = nn.Linear(self.fc3.out_features, self.fc3.out_features * 2)
        self.fc5 = nn.Linear(self.fc4.out_features, self.fc4.out_features * 2)
        self.fc6 = nn.Linear(self.fc5.out_features, self.fc5.out_features * 2)
        self.fc7 = nn.Linear(self.fc6.out_features, g_output_dim)

    # forward method
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = F.leaky_relu(self.fc4(x), 0.2)
        x = F.leaky_relu(self.fc5(x), 0.2)
        x = F.leaky_relu(self.fc6(x), 0.2)
        return torch.tanh(self.fc7(x))