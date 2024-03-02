# Code based on https://github.com/amazon-science/object-centric-learning-framework/blob/main/ocl

import torch
from torch import nn

class MlpDecoder(nn.Module):
    """Decoder that takes object representations and reconstructs patches.

    Args:
        object_dim: Dimension of objects representations.
        output_dim: Dimension of each patch.
        num_patches: Number of patches P to reconstruct.
        hidden_features: Dimension of hidden layers.
    """

    def __init__(self, object_dim, output_dim, num_patches, hidden_features = 2048):
        super().__init__()
        self.output_dim = output_dim
        self.num_patches = num_patches
        
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, object_dim) * 0.02)
        self.decoder = build_mlp(object_dim, output_dim + 1, hidden_features)

    def forward(self, encoder_output):

        initial_shape = encoder_output.shape[:-1]
        encoder_output = encoder_output.flatten(0, -2)

        encoder_output = encoder_output.unsqueeze(1).expand(-1, self.num_patches, -1)

        # Simple learned additive embedding as in ViT
        object_features = encoder_output + self.pos_embed

        output = self.decoder(object_features)
        output = output.unflatten(0, initial_shape)

        # Split out alpha channel and normalize over slots.
        decoded_patches, alpha = output.split([self.output_dim, 1], dim=-1)
        alpha = alpha.softmax(dim=-3)

        reconstruction = torch.sum(decoded_patches * alpha, dim=-3)
        masks = alpha.squeeze(-1)
        
        return reconstruction, masks
    
    
def build_mlp(input_dim = int, output_dim = int, hidden_features = 2048, n_hidden_layers = 3):
    
    layers = []
    current_dim = input_dim
    features = [hidden_features]*n_hidden_layers

    for n_features in features:
        layers.append(nn.Linear(current_dim, n_features))
        nn.init.zeros_(layers[-1].bias)
        layers.append(nn.ReLU(inplace=True))
        current_dim = n_features

    layers.append(nn.Linear(current_dim, output_dim))
    nn.init.zeros_(layers[-1].bias)

    return nn.Sequential(*layers)