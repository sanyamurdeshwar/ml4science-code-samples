from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class VAEConv(nn.Module):
    def __init__(self, zdim: int = 32):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(1, 32, 3, 2, 1), nn.ReLU(True),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(True),
            nn.Flatten(),
        )
        self.fc_mu = nn.Linear(64 * 7 * 7, zdim)
        self.fc_logvar = nn.Linear(64 * 7 * 7, zdim)

        self.dec_fc = nn.Linear(zdim, 256 * 7 * 7)
        self.dec = nn.Sequential(
            nn.Unflatten(1, (256, 7, 7)), nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(True),
            nn.Conv2d(64, 1, 3, 1, 1),
            nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.enc(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparam(self, mu, logvar):
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.dec(self.dec_fc(z))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparam(mu, logvar)
        xrec = self.decode(z)
        return xrec, mu, logvar


class VAEMlp(nn.Module):
    def __init__(self, indim: int = 2, zdim: int = 32):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(indim, 512), nn.ReLU(True),
            nn.Linear(512, 512), nn.ReLU(True),
        )
        self.fc_mu = nn.Linear(512, zdim)
        self.fc_logvar = nn.Linear(512, zdim)

        self.dec = nn.Sequential(
            nn.Linear(zdim, 512), nn.ReLU(True),
            nn.Linear(512, 512), nn.ReLU(True),
            nn.Linear(512, indim),
        )

    def encode(self, x):
        h = self.enc(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparam(self, mu, logvar):
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.dec(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparam(mu, logvar)
        xrec = self.decode(z)
        return xrec, mu, logvar


class VAEManifoldMlp(nn.Module):
    """
    Manifold-aware VAE:
    - encodes x in R^3
    - decoder outputs 2 parameters which are mapped to the manifold
    """
    def __init__(self, indim: int = 3, zdim: int = 64, manifold_type: str = "Sphere", R: float = 3.2):
        super().__init__()
        self.manifold_type = manifold_type
        self.R = float(R)

        self.enc = nn.Sequential(
            nn.Linear(indim, 512), nn.ReLU(True),
            nn.Linear(512, 512), nn.ReLU(True),
        )
        self.fc_mu = nn.Linear(512, zdim)
        self.fc_logvar = nn.Linear(512, zdim)

        self.dec = nn.Sequential(
            nn.Linear(zdim, 512), nn.ReLU(True),
            nn.Linear(512, 512), nn.ReLU(True),
            nn.Linear(512, 2),
        )

    def encode(self, x):
        h = self.enc(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparam(self, mu, logvar):
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        coords = self.dec(z)  # (B,2)

        if self.manifold_type.lower() == "sphere":
            theta = torch.sigmoid(coords[:, 0]) * (2.0 * np.pi)
            mu = torch.tanh(coords[:, 1])  # [-1, 1]
            phi = torch.acos(torch.clamp(mu, -0.999, 0.999))
            x = torch.sin(phi) * torch.cos(theta)
            y = torch.sin(phi) * torch.sin(theta)
            z_out = torch.cos(phi)
            return torch.stack([x, y, z_out], dim=1)

        if self.manifold_type.lower() == "hyperboloid":
            theta = torch.sigmoid(coords[:, 0]) * (2.0 * np.pi)
            r = F.softplus(coords[:, 1]) * self.R
            sinh_r = torch.sinh(r)
            cosh_r = torch.cosh(r)
            x1 = sinh_r * torch.cos(theta)
            x2 = sinh_r * torch.sin(theta)
            x0 = cosh_r
            return torch.stack([x1, x2, x0], dim=1)

        raise ValueError(f"Unknown manifold_type: {self.manifold_type}")

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparam(mu, logvar)
        xrec = self.decode(z)
        return xrec, mu, logvar
