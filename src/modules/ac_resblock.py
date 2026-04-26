import torch.nn as nn
from blocks.acb import ACB


class ACResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.conv = nn.Sequential(
            ACB(channels, channels),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return x + self.conv(x)
