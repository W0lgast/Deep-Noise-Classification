"""
This contains a class containing multiple trained CNNs,
logits are calculated then averaged.

Kipp McAdam Freud, Stoil Ganev
19/12/2019
"""
# --------------------------------------------------------------

import torch
import torch.backends.cudnn
import numpy as np
from torch import nn, optim
from torch.nn import functional as F
import pickle

from util.message import message
import util.utilities as ut

# --------------------------------------------------------------

class CLateFusionCNN(nn.Module):
    def __init__(self,
                 cnn_list: list):
        super().__init__()
        if len(cnn_list) == 1:
            message.logError("List of CNNs to fuse must have length greater than 1.",
                             "CLateFusionCNN::__init__")
            ut.exit(0)
        self.cnn_models = cnn_list
        for cnn in self.cnn_models:
            cnn.eval()

    def forward(self, images: torch.Tensor) -> torch.Tensor:

        all_logits = []
        for i, cnn in enumerate(self.cnn_models):
            all_logits.append(cnn(images[:,i]))
        return torch.div(torch.add(*all_logits), len(all_logits))

    def save(self, filename: str):
        outfile = open(filename, 'wb')
        pickle.dump(self, outfile)
        outfile.close()