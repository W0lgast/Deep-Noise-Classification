"""
This contains a class representing the network used in paper.pdf

Kipp McAdam Freud, Stoil Ganev
19/12/2019
"""
# --------------------------------------------------------------

import torch
import torch.backends.cudnn
import numpy as np
from torch import nn, optim
from torch.nn import functional as F

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
            stride=1, # We believe stride = (2,2), which is written in the paper, is erroneous.
            padding=(1,1), #or is it 0?!
        )

        self.conv2 = nn.Conv2d(
            in_channels=self.conv1.out_channels,
            out_channels=32,
            kernel_size=(3, 3),
            stride=1, # We believe stride = (2,2), which is written in the paper, is erroneous.
            padding=(1, 1),  # or is it zero?!
        )

        self.conv3 = nn.Conv2d(
            in_channels=self.conv2.out_channels,
            out_channels=64,
            kernel_size=(3, 3),
            stride=1, # We believe stride = (2,2), which is written in the paper, is erroneous.
            padding=(1, 1),  # or is it zero?!
        )

        self.conv4 = nn.Conv2d(
            in_channels=self.conv3.out_channels,
            out_channels=64,
            kernel_size=(3, 3),
            stride=1, # We believe stride = (2,2), which is written in the paper, is erroneous.
            padding=(1, 1),  # or is it zero?!
        )

        # This is the number of pixels of the image when sent to the first fully connected layer.
        size_flat = int(self.conv4.out_channels * np.ceil(height/4.0) * np.ceil(width/4.0))

        self.fc1 = nn.Linear(size_flat, 1024) #INP CHANNELS PROBABLY WRONG!

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

        self.max_pool_1 = nn.MaxPool2d(kernel_size=(2, 2),
                                       stride=(2, 2),
                                       padding=(1, 1))

        # Second max pool not explicitly mentioned in paper, but we're using it to compensate for an
        # otherwise unexplained dimensionality reduction in terrible paper.
        self.max_pool_2 = nn.MaxPool2d(kernel_size=(2, 2),
                                       stride=(2, 2),
                                       padding=(1, 1))

        self.initialise_layer(self.conv1)
        self.initialise_layer(self.conv2)
        self.initialise_layer(self.conv3)
        self.initialise_layer(self.conv4)
        self.initialise_layer(self.fc1)
        self.initialise_layer(self.fc2)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = F.relu(
            self.norm2d1(
                self.conv1(images)
            )
        )

        x = self.max_pool_1(
            F.relu(
                self.norm2d2(
                    self.conv2(
                        self.dropout(x)
                    )
                )
            )
        )

        x = F.relu(
            self.norm2d3(
                self.conv3(x)
            )
        )

        x = self.max_pool_2(
            F.relu(
                self.norm2d4(
                    self.conv4(
                        self.dropout(x)
                    )
                )
            )
        )

        x = torch.flatten(x, start_dim=1) #DO I USE THIS?

        x = F.sigmoid(
            self.fc1(
                x
            )
        )

        x = F.softmax(
            self.fc2(
                self.dropout(x)
            )
        )

        return x

    @staticmethod
    def initialise_layer(layer):
        if hasattr(layer, "bias"):
            nn.init.zeros_(layer.bias)
        if hasattr(layer, "weight"):
            nn.init.kaiming_normal_(layer.weight)
