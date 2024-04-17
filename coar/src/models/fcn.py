import sys
import copy
import torch
import torchvision
from torch import optim, nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

def get_fcn(input_dim, hidden_dim, output_dim, num_layers,
            use_kaiming_init=False, activation_func=nn.ReLU,
            use_activation=True, use_batchnorm=False,
            use_bias=True, dropout_prob=0):
    """
    - input_dim: input dimension
    - hidden_dim: hidden dimension
    - output_dim: output dimension
    - num_layers: number of hidden layers (0 -> linear model)
    - use_kaiming_init: use kaiming initalization instead of default pytorch
    - activation func: nn. activation function (default is relu)
    - use_activation: if false, linear neural net
    - use_batchnorm: use batch normalization
    - dropout prob: dropout probability (default is 0 / disabled)
    """
    def kaiming_init(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight.data)
            nn.init.kaiming_uniform_(m.bias.data)

    def _get_layers(input_dim):
        layers = [nn.Linear(input_dim, hidden_dim, bias=use_bias)]
        if use_activation: layers.append(activation_func())
        if dropout_prob > 0: layers.append(nn.Dropout(dropout_prob))
        if use_batchnorm: layers.append(nn.BatchNorm1d(hidden_dim))
        return layers

    all_layers = []

    for layer_idx in range(num_layers):
        i_dim = input_dim if layer_idx==0 else hidden_dim
        layers = _get_layers(i_dim)
        all_layers.extend(layers)

    i_dim = input_dim if num_layers==0 else hidden_dim
    all_layers.append(nn.Linear(i_dim, output_dim))
    model = nn.Sequential(*all_layers)

    if use_kaiming_init:
        model.apply(kaiming_init)

    return model

def get_linear_model(input_dim, output_dim):
    return get_fcn(input_dim, 0, output_dim, 0)