import torch
import torch.nn as nn
from res_net import *

GEN_SIZE = 128
DIS_SIZE = 128

class Generator(nn.Module):
    def __init__(self, z_dim):
        super(Generator, self).__init__()
        self.z_dim = z_dim

        self.dense = nn.Linear(self.z_dim, 4 * 4 * GEN_SIZE)
        self.final = nn.Conv2d(GEN_SIZE, channels, 3, stride = 1, padding = 1)

        # Xavier Initialization
        nn.init.xavier_uniform(self.dense.weight.data, 1.0)
        nn.init.xavier_uniform(self.final.weight.data, 1.0)

        self.model = nn.Sequential(
                        ResBlockGen(GEN_SIZE, GEN_SIZE, stride = 2),
                        ResBlockGen(GEN_SIZE, GEN_SIZE, stride = 2),
                        ResBlockGen(GEN_SIZE, GEN_SIZE, stride = 2),
                        nn.BatchNorm2d(GEN_SIZE),
                        nn.ReLU(),
                        self.final,
                        nn.Tanh()
                    )
    def forward(self, z):
        return self.model(self.dense(z).view(-1, GEN_SIZE, 4, 4))

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
                        FirstResNetDis(channels, DIS_SIZE, stride = 2),
                        ResBlockDis(DIS_SIZE, DIS_SIZE, stride = 2),
                        ResBlockDis(DIS_SIZE, DIS_SIZE),
                        ResBlockDis(DIS_SIZE, DIS_SIZE),
                        nn.ReLU(),
                        nn.AvgPool2d(8)
                    )
        self.fc_base = nn.Linear(DIS_SIZE, 1)
        nn.init.xavier_uniform(self.fc_base.weight.data, 1.0)
        self.fc = nn.Sequential(
                    nn.BatchNorm1d(DIS_SIZE),
                    self.fc_base
                )

    def forward(self, x):
        # print('X SHAPE:', x.shape)
        x = self.model(x).view(-1, DIS_SIZE)
        # print('Res Done', x.shape)
        x = self.fc(x)
        return x
