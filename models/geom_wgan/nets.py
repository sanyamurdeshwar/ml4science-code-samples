from __future__ import annotations
import torch.nn as nn

class GeneratorMNIST(nn.Module):
    def __init__(self, z_dim: int = 100):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 128*7*7),
            nn.BatchNorm1d(128*7*7),
            nn.ReLU(True),
            nn.Unflatten(1, (128, 7, 7)),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 1, 3, 1, 1),
            nn.Tanh(),
        )

    def forward(self, z): return self.net(z)

class CriticMNIST(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(128*4*4, 1),
        )

    def forward(self, x): return self.net(x).view(-1)

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden, depth):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden), nn.LeakyReLU(0.2, inplace=True)]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.LeakyReLU(0.2, inplace=True)]
        layers += [nn.Linear(hidden, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x): return self.net(x)

class Generator2D(nn.Module):
    def __init__(self, z_dim: int = 64, hidden: int = 512, depth: int = 4):
        super().__init__()
        self.net = MLP(z_dim, 2, hidden, depth)

    def forward(self, z): return self.net(z)

class Critic2D(nn.Module):
    def __init__(self, hidden: int = 512, depth: int = 4):
        super().__init__()
        self.net = MLP(2, 1, hidden, depth)

    def forward(self, x): return self.net(x).view(-1)
