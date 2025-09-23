# Data Classification with FashionMnist
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Loading data
train_data = datasets.FashionMNIST(
    root="./train",
    train=True,
    download=True,
    transform=transforms.ToTensor()
)
test_data = datasets.FashionMNIST(
    root="./test",
    train=False,
    download=True,
    transform=transforms.ToTensor()
)

# Setting up the device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"{device} is available")

# Defining model