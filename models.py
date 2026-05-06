import torch
import torch.nn as nn
from utils import LoRAParameterization


class NeuralNet(nn.Module):
    def __init__(self, hidden_size_1=1000, hidden_size_2=2000) -> None:
        super(NeuralNet, self).__init__()
        self.linear1 = nn.Linear(28 * 28, hidden_size_1)
        self.linear2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.linear3 = nn.Linear(hidden_size_2, 10)
        self.relu = nn.ReLU()

        # initalize lora modifications
        self.add_lora_parameterizations()

    def forward(self, img):
        # forward pass through network
        x = img.view(-1, 28 * 28)
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.linear3(x)
        return x

    def add_lora_parameterizations(self):

        # adds lora parameterization to each linear layer

        # retrive the device to place LoRA parameteres on the same device as model
        device = next(self.parameters()).device

        # add LoRA to each linear layer with default settings
        LoRAParameterization.add_to_layer(self.linear1, device)
        LoRAParameterization.add_to_layer(self.linear2, device)
        LoRAParameterization.add_to_layer(self.linear3, device)
