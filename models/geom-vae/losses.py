from __future__ import annotations
import torch
import torch.nn.functional as F

def kld_loss(mu, logvar):
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

def sphere_rec_loss(x_rec, x_true, eps=1e-7):
    # x_rec, x_true: (B,3) on S^2
    dot = (x_rec * x_true).sum(dim=1).clamp(-1 + eps, 1 - eps)
    return (2.0 * (1.0 - dot)).mean()

def hyper_rec_loss_x1x2x0(x_rec, x_true, eps=1e-7):
    # coords are [x1,x2,x0] with x0 time-like (last)
    u0, u1, u2 = x_rec[:, 2], x_rec[:, 0], x_rec[:, 1]
    v0, v1, v2 = x_true[:, 2], x_true[:, 0], x_true[:, 1]
    mink = -(u0 * v0) + (u1 * v1) + (u2 * v2)
    arg = (-mink).clamp_min(1.0 + eps)  # acosh domain
    d = torch.acosh(arg)
    return (d * d).mean()

def mse_rec_loss(x_rec, x_true):
    return F.mse_loss(x_rec, x_true, reduction="mean")
