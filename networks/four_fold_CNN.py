"""
This contains a class representing the network used in paper.pdf

Kipp McAdam Freud, Stoil Ganev
19/12/2019
"""
# --------------------------------------------------------------

from multiprocessing import cpu_count
from typing import Union, NamedTuple
import torch
import torch.backends.cudnn
import numpy as np
from torch import nn, optim
from torch.nn import functional as F
import torchvision.datasets
from torchvision.transforms import Compose
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from util.message import message
import util.utilities as ut
from util.data_proc import ImageShape

# --------------------------------------------------------------

class CFourFoldCNN(nn.Module):
    def __init__(self,
                 height: int,
                 width: int,
                 channels: int,
                 class_count: int,
                 dropout: float):
        super().__init__()
        self.input_shape = ImageShape(height=height, width=width, channels=channels)
        self.class_count = class_count
        self.dropout = nn.Dropout(p=dropout)

        self.conv1 = nn.Conv2d(
            in_channels=self.input_shape.channels,
            out_channels=32,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(1, 1), #or is it zero?!
        )

        self.conv2 = nn.Conv2d(
            in_channels=self.conv1.out_channels,
            out_channels=32,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(1, 1),  # or is it zero?!
        )

        self.conv3 = nn.Conv2d(
            in_channels=self.conv2.out_channels,
            out_channels=64,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(1, 1),  # or is it zero?!
        )

        self.conv4 = nn.Conv2d(
            in_channels=self.conv3.out_channels,
            out_channels=64,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(1, 1),  # or is it zero?!
        )

        self.fc1 = nn.Linear(self.conv4.out_channels, 1024) #INP CHANNELS PROBABLY WRONG!

        self.fc2 = nn.Linear(1024, 10)

        self.norm2d1 = nn.BatchNorm2d(
            num_features=32
        )
        self.norm2d2 = nn.BatchNorm2d(
            num_features=32
        )
        self.norm2d3 = nn.BatchNorm2d(
            num_features=64
        )
        self.norm2d4 = nn.BatchNorm2d(
            num_features=64
        )

        self.initialise_layer(self.conv1)
        self.initialise_layer(self.conv2)
        self.initialise_layer(self.conv3)
        self.initialise_layer(self.conv4)
        self.initialise_layer(self.fc1)
        self.initialise_layer(self.fc2)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = F.relu( self.norm2d1(
            self.conv1(images)
            )
        )

        x = F.relu( self.norm2d2(
            self.conv2(x)
            )
        )

        x = F.relu(self.norm2d3(
            self.conv3(x)
            )
        )

        x = F.relu(self.norm2d4(
            self.conv4(x)
            )
        )

        #x = torch.flatten(x, start_dim=1) #DO I USE THIS?

        x = F.sigmoid(
            self.fc1(
                x
            )
        )

        x = F.softmax(
            self.fc2(
                x
            )
        )

        return x

    @staticmethod
    def initialise_layer(layer):
        if hasattr(layer, "bias"):
            nn.init.zeros_(layer.bias)
        if hasattr(layer, "weight"):
            nn.init.kaiming_normal_(layer.weight)