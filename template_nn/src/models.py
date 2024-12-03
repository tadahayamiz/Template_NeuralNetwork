# -*- coding: utf-8 -*-
"""
Created on Fri 29 15:46:32 2022

ResNet50 architecture

@author: tadahaya
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class Block(nn.Module):
    """
    A residual block consisting of three convolutional layers with skip connections.

    Args:
        channel_in (int): Number of input channels.
        channel_out (int): Number of output channels.
    """
    def __init__(self, channel_in, channel_out):
        super().__init__()
        channel = channel_out // 4 # Bottleneck dimension

        # 1 x 1 convolution
        self.conv1 = nn.Conv2d(channel_in, channel, kernel_size=(1,1))
        self.bn1 = nn.BatchNorm2d(channel)
        self.relu1 = nn.ReLU()

        # 3 x 3 convolution
        self.conv2 = nn.Conv2d(channel, channel, kernel_size=(3,3), padding=1)
        self.bn2 = nn.BatchNorm2d(channel)
        self.relu2 = nn.ReLU()

        # 1 x 1 convolution
        self.conv3 = nn.Conv2d(channel, channel_out, kernel_size=(1,1), padding=0)
        self.bn3 = nn.BatchNorm2d(channel_out)

        # prep for skip connection
        self.shortcut = self._shortcut(channel_in, channel_out)
        self.relu3 = nn.ReLU()

    def forward(self,x):
        """
        Forward pass through the residual block.
        
        Args:
            x (Tensor): Input tensor of shape (batch_size, channel_in, height, width).

        Returns:
            Tensor: Output tensor of shape (batch_size, channel_out, height, width).
        """
        h = self.conv1(x)
        h = self.bn1(h)
        h = self.relu1(h)
        h = self.conv2(h)
        h = self.bn2(h)
        h = self.relu2(h)
        h = self.conv3(h)
        h = self.bn3(h)
        shortcut = self.shortcut(x)
        y = self.relu3(h + shortcut) # skip connection
        return y

    def _shortcut(self, channel_in, channel_out):
        """
        Defines the shortcut connection. If the input and output channel counts differ,
        applies a 1x1 convolution to match dimensions.

        Args:
            channel_in (int): Input channel count.
            channel_out (int): Output channel count.

        Returns:
            Callable: Either identity mapping or projection.
        """
        if channel_in != channel_out:
            return nn.Conv2d(channel_in, channel_out, kernel_size=(1, 1), stride=1)
        return nn.Identity()  # Direct pass-through if dimensions match

    def _projection(self, channel_in, channel_out):
        """
        Projects the input to the output channel dimensions using a 1x1 convolution.

        Args:
            channel_in (int): Input channel count.
            channel_out (int): Output channel count.

        Returns:
            nn.Conv2d: Projection layer.
        """
        return nn.Conv2d(channel_in, channel_out,
                         kernel_size=(1, 1),
                         padding=0)

class MyNet(nn.Module):
    """
    A ResNet-like neural network for classification tasks.

    Args:
        output_dim (int): Number of output dimensions for the final classification layer.
    """
    def __init__(self, output_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=(7,7), stride=(2,2), padding=3
            )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(
            kernel_size=(3,3), stride=(2,2), padding=1
            )
        
        # block 1
        self.block0 = self._building_block(256, channel_in=64)
        self.block1 = nn.ModuleList(
            [self._building_block(256) for _ in range(2)]
        )
        self.conv2 = nn.Conv2d(256, 512, kernel_size=(1,1), stride=(2,2))

        # block 2
        self.block2 = nn.ModuleList(
            [self._building_block(512) for _ in range(4)]
        )
        self.conv3 = nn.Conv2d(512, 1024, kernel_size=(1,1), stride=(2,2))

        # block 3
        self.block3 = nn.ModuleList(
            [self._building_block(1024) for _ in range(6)]
        )
        self.conv4 = nn.Conv2d(1024, 2048, kernel_size=(1,1), stride=(2,2))

        # block 4
        self.block4 = nn.ModuleList(
            [self._building_block(2048) for _ in range(3)]
        )

        self.avg_pool = GlobalAvgPool2d()
        self.fc = nn.Linear(2048, 512)
        self.out = nn.Linear(512, output_dim)

    def forward(self,x):
        """
        Forward pass through the network.
        
        Args:
            x (Tensor): Input tensor of shape (batch_size, 3, height, width).

        Returns:
            Tensor: Output tensor of shape (batch_size, output_dim).
        """
        h = self.conv1(x) # Shape: (batch_size, C, H, W)
        h = self.bn1(h)
        h = self.relu1(h)
        h = self.pool1(h)
        h = self.block0(h)
        for block in self.block1:
            h = block(h)
        h = self.conv2(h) # Shape: (batch_size, C, H, W)
        for block in self.block2:
            h = block(h)
        h = self.conv3(h) # Shape: (batch_size, C, H, W)
        for block in self.block3:
            h = block(h)
        h = self.conv4(h) # Shape: (batch_size, C, H, W)
        for block in self.block4:
            h = block(h)
        h = self.avg_pool(h) # Shape: (batch_size, C)
        h = self.fc(h) # Shape: (batch_size, hidden_dim)
        h = torch.relu(h)
        h = self.out(h) # Shape: (batch_size, output_dim)
        return h

    def _building_block(self, channel_out, channel_in=None):
        """
        Helper function to create a residual block.

        Args:
            channel_out (int): Number of output channels.
            channel_in (int, optional): Number of input channels. Defaults to channel_out.

        Returns:
            Block: A residual block instance.
        """
        if channel_in is None:
            channel_in = channel_out
        return Block(channel_in,channel_out)


class GlobalAvgPool2d(nn.Module):
    """
    Global average pooling layer.
    """
    def __init__(self):
        super().__init__()

    def forward(self,x):
        """
        Applies global average pooling.

        Args:
            x (Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            Tensor: Output tensor of shape (batch_size, channels).
        """    
        return F.avg_pool2d(x, kernel_size=x.size()[2:]).view(x.size(0), -1)