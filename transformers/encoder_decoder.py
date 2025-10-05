import torch
from torch import nn
import os
import sys

current_dir = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.append(grandparent_dir)

from transformers.attention import Attention


class TransformerEncoder(nn.Module):
    def __init__(self, in_channel: int, out_channel: int):
        super(TransformerEncoder, self).__init__()
        self.self_attention = Attention(in_channel=in_channel, out_channel=out_channel)
        self.feedforward = nn.Linear(in_features=in_channel, out_features=out_channel)
        self.layernorm = nn.LayerNorm()

    def forward(x):
        #

        return z
