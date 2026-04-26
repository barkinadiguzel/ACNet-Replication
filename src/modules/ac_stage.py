import torch.nn as nn
from modules.ac_resblock import ACResBlock


class ACStage(nn.Module):
    def __init__(self, channels, num_blocks):
        super().__init__()

        self.blocks = nn.Sequential(
            *[ACResBlock(channels) for _ in range(num_blocks)]
        )

    def forward(self, x):
        return self.blocks(x)
