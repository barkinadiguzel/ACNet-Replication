import torch
import torch.nn as nn
from modules.ac_stage import ACStage


class ACNet(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.stage1 = ACStage(64, 2)
        self.down1 = nn.Conv2d(64, 128, 3, stride=2, padding=1)

        self.stage2 = ACStage(128, 2)
        self.down2 = nn.Conv2d(128, 256, 3, stride=2, padding=1)

        self.stage3 = ACStage(256, 2)

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.stem(x)

        x = self.stage1(x)
        x = self.down1(x)

        x = self.stage2(x)
        x = self.down2(x)

        x = self.stage3(x)

        x = self.pool(x)
        x = x.flatten(1)

        x = self.classifier(x)
        return x
