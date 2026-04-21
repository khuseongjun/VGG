import torch
import torch.nn as nn

class VGG(nn.Module):
    def __init__(self, channels=[32, 64, 128, 256, 512], pool_stride=2):
        super(VGG, self).__init__()

        c1, c2, c3, c4, c5 = channels

        self.block1 = nn.Sequential(
            nn.Conv2d(3, c1, 3, 1, 1),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),

            nn.Conv2d(c1, c2, 3, 1, 1),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=pool_stride)
        )

        self.block2_1 = nn.Sequential(
            nn.Conv2d(c2, c2, 3, 1, 1),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),

            nn.Conv2d(c2, c3, 3, 1, 1),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
        )

        self.block2_2 = nn.Sequential(
            nn.Conv2d(c3, c3, 3, 1, 1),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),

            nn.Conv2d(c3, c4, 3, 1, 1),
            nn.BatchNorm2d(c4),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=pool_stride)
        )

        self.block3_1 = nn.Sequential(
            nn.Conv2d(c4, c4, 3, 1, 1),
            nn.BatchNorm2d(c4),
            nn.ReLU(inplace=True),

            nn.Conv2d(c4, c5, 3, 1, 1),
            nn.BatchNorm2d(c5),
            nn.ReLU(inplace=True),
        )

        self.block3_2 = nn.Sequential(
            nn.Conv2d(c5, c5, 3, 1, 1),
            nn.BatchNorm2d(c5),
            nn.ReLU(inplace=True),

            nn.Conv2d(c5, c5, 3, 1, 1),
            nn.BatchNorm2d(c5),
            nn.ReLU(inplace=True),
        )

        self.block3_3 = nn.Sequential(
            nn.Conv2d(c5, c5, 3, 1, 1),
            nn.BatchNorm2d(c5),
            nn.ReLU(inplace=True),

            nn.Conv2d(c5, c5, 3, 1, 1),
            nn.BatchNorm2d(c5),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=pool_stride)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))

        self.fc = nn.Sequential(
            nn.Linear(c5 * 4 * 4, 1024),
            nn.ReLU(inplace=True),

            nn.Linear(1024, 128),
            nn.ReLU(inplace=True),

            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2_1(x)
        x = self.block2_2(x)
        x = self.block3_1(x)
        x = self.block3_2(x)
        x = self.block3_3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x