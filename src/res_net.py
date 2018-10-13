import torch
from torch import nn
import torch.nn.functional as F

channels = 3
class ResBlockGen(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1):
        super(ResBlockGen, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding = 1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding = 1)

        # Batch Normalization
        self.bn1   = nn.BatchNorm2d(in_channels)
        self.bn2   = nn.BatchNorm2d(out_channels)

        # Initialization
        # Xavier Initialization
        nn.init.xavier_uniform(self.conv1.weight.data, 1.0)
        nn.init.xavier_uniform(self.conv2.weight.data, 1.0)

        self.model = nn.Sequential (
                        self.bn1,
                        nn.ReLU(),
                        nn.Upsample(scale_factor = 2),
                        self.conv1,
                        self.bn2,
                        nn.ReLU(),
                        self.conv2
                    )
        self.bypass = nn.Sequential()
        if stride != 1:
            self.bypass = nn.Upsample(scale_factor = 2)

    def forward(self, x):
        return self.model(x) + self.bypass(x)

class ResBlockDis(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1):
        super(ResBlockDis, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding = 1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding = 1)

        # Batch Normalization
        self.bn1   = nn.BatchNorm2d(in_channels)
        self.bn2   = nn.BatchNorm2d(out_channels)
        self.bn3   = nn.BatchNorm2d(out_channels)

        # Initialization
        # Xavier Initialization
        nn.init.xavier_uniform(self.conv1.weight.data, 1.0)
        nn.init.xavier_uniform(self.conv2.weight.data, 1.0)

        self.model_base = nn.Sequential (
                            nn.ReLU(),
                            self.bn1,
                            self.conv1,
                            nn.ReLU(),
                            self.bn2,
                            self.conv2
                        )

        self.bypass = nn.Sequential()
        if stride == 1:
            self.model = self.model_base
        else:
            self.model = nn.Sequential(
                            self.model_base,
                            nn.AvgPool2d(2, stride = stride, padding = 0)
                        )
            # add Bypass
            self.bypass_conv = nn.Conv2d(in_channels, out_channels, 1, 1, padding = 0)
            self.bypass = nn.Sequential (
                            self.bn3,
                            self.bypass_conv,
                            nn.AvgPool2d(2, stride = stride, padding = 0)
                        )
            # Xavier Initialization
            nn.init.xavier_uniform(self.bypass_conv.weight.data, 1.4142)

    def forward(self, x):
        # print('SECOND BLOCK ONWARDS')
        return self.model(x) + self.bypass(x)

class FirstResNetDis(nn.Module):

    def __init__(self, in_channels, out_channels, stride = 1):
        super(FirstResNetDis, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding = 1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding = 1)
        self.bypass_conv = nn.Conv2d(in_channels, out_channels, 1, 1, padding = 0)

        # Batch Normalization
        self.bn1   = nn.BatchNorm2d(in_channels)
        self.bn2   = nn.BatchNorm2d(out_channels)
        self.bn3   = nn.BatchNorm2d(in_channels)

        # Xavier Initialization
        nn.init.xavier_uniform(self.conv1.weight.data, 1.0)
        nn.init.xavier_uniform(self.conv2.weight.data, 1.0)
        nn.init.xavier_uniform(self.bypass_conv.weight.data, 1.4142)

        self.model = nn.Sequential(
                self.bn1,
                self.conv1,
                nn.ReLU(),
                self.bn2,
                self.conv2,
                nn.AvgPool2d(2)
            )

        self.bypass = nn.Sequential(
                        nn.AvgPool2d(2),
                        self.bn3,
                        self.bypass_conv
                    )
    def forward(self, x):
        # print('IN FIRST BLOCK')
        return self.model(x) + self.bypass(x)
