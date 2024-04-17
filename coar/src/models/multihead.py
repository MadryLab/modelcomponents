import os 
import sys 
from collections import defaultdict
import numpy as np 

import torch 
from torch import nn 
import torch.nn.functional as F 
import torchvision 

class MultiHeadModel(nn.Module):
    """
    Models with single base component 
    and one or more head components 
    """

    def __init__(self, base_model, heads):
        """
        - base_model: base_model (nn.Module instance)
        - heads: list of head models (nn.Module instances)
        """
        assert isinstance(base_model, nn.Module)
        assert isinstance(heads, list)
        assert all([isinstance(h, nn.Module) for h in heads])

        # setup
        super().__init__()
        self.base_model = base_model
        self.num_heads = len(heads)
        
        # init heads / register models w. self
        self.heads = [] 
        for idx, head in enumerate(heads):
            setattr(self, f'head{idx}', head)
            self.heads.append(getattr(self, f'head{idx}'))

    def forward(self, x):
        # return num_heads x batch_size x num_classes tensor
        base_out = self.base_model(x)
        head_outs = [self.heads[idx](base_out) for idx in range(self.num_heads)]
        return head_outs
        
    def extract_single_head_model(self, head_idx):
        assert 0 <= head_idx < self.num_heads
        return SingleHeadModel(self.base_model, self.heads[head_idx], squeeze=True)

class Ensemble(MultiHeadModel):

    def __init__(self, models):
        identity = nn.Identity()
        super().__init__(identity, models)
        self.models = self.heads

class SingleHeadModel(MultiHeadModel):
    """
    Multi-head wrapper for standard torch models
    - model: torch nn.Module
    - squeeze: remove first dim of multi-head output
               set squeeze=False to use multihead trainer
               set squeeze=True to use it as standard torch model
    """

    def __init__(self, base_model, head, squeeze=True):
        super().__init__(base_model, [head])
        self.squeeze = squeeze

    def forward(self, x):
        out = super().forward(x)
        if self.squeeze: 
            return out[0]
        return out