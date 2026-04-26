import torch
import torch.nn as nn
import torch.nn.functional as F


class ACB(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False):
        super().__init__()

        self.conv3x3 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=bias)
        self.conv1x3 = nn.Conv2d(in_channels, out_channels, (1, 3), padding=(0, 1), bias=bias)
        self.conv3x1 = nn.Conv2d(in_channels, out_channels, (3, 1), padding=(1, 0), bias=bias)

    def forward(self, x):
        return self.conv3x3(x) + self.conv1x3(x) + self.conv3x1(x)
