import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from models import NeuralNet
from torch.utils.data import DataLoader, Subset
from utils import count_trainable_params, enable_disable_lora, test, train


def main():
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.137,), (0.3081,))]
    )
