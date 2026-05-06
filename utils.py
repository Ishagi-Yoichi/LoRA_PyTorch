import torch
import torch.nn as nn
from tqdm import tqdm


class LoRAParameterization(nn.Module):
    # Implements LoRA to modify layer weights during training

    def __init__(self, features_in, features_out, rank=1, alpha=1, device="cpu"):
        super().__init__()
        # initialize low-rank matrices A and B
        # a random guassian initialization for A and zero for B,
        self.lora_A = nn.Parameter(torch.randn(rank, features_out, device=device))
        self.lora_B = nn.Parameter(torch.zeros(features_in, rank, device=device))

        # we then scale ∆Wx by α/r , where α is a constant in r.
        # when optimiziming with Adam, tuning α is roughly the same as tuning the learning rate if we scale the initailization appropriately.
        # As a result, α is set to the first r we try and do not tune it.
        # this scaling helps to reduce the need to reduce hyperparamters when we vary r.
        self.scale = alpha / rank  # scale delta weights
        self.enabled = True  # Controls if LoRA is applied or original weights are used

    def forward(self, original_weights):
        # Modify original weights based on LoRA parameters if enabled
        if self.enabled:
            delta_weights = torch.matmul(self.lora_B, self.lora_A).view(
                original_weights.shape
            )
            return original_weights + delta_weights * self.scale
        return original_weights

    @staticmethod
    def add_to_layer(layer, device, rank=1, alpha=1):
        # Capture input/output features of the layer to size LoRA matrices appropriately
        features_in, features_out = layer.weights.shape
        parameterization = LoRAParameterization(
            features_in, features_out, rank, alpha, device
        )
        nn.utils.parametrize.register_parametrization(layer, "weight", parameterization)
