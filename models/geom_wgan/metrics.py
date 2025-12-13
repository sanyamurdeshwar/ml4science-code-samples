from __future__ import annotations
import math
import torch
import torch.nn.functional as F

@torch.no_grad()
def sliced_wasserstein_2d(x: torch.Tensor, y: torch.Tensor, k: int = 512, device=None) -> float:
    device = device or x.device
    x = x.to(device); y = y.to(device)
    theta = torch.rand(k, device=device) * 2 * math.pi
    dirs = torch.stack([torch.cos(theta), torch.sin(theta)], dim=1)  # [k,2]
    x_proj = (x @ dirs.t()).sort(dim=0).values
    y_proj = (y @ dirs.t()).sort(dim=0).values
    return (x_proj - y_proj).abs().mean().item()

@torch.no_grad()
def sliced_wasserstein_images(x: torch.Tensor, y: torch.Tensor, k: int = 256, device=None) -> float:
    device = device or x.device
    n = min(x.size(0), y.size(0))
    x = x[:n].reshape(n, -1).to(device)
    y = y[:n].reshape(n, -1).to(device)
    dirs = F.normalize(torch.randn(k, x.size(1), device=device), dim=1)
    x_proj = (x @ dirs.t()).sort(dim=0).values
    y_proj = (y @ dirs.t()).sort(dim=0).values
    return (x_proj - y_proj).abs().mean().item()

def gradient_penalty(D, real, fake, lambda_gp: float = 10.0):
    bs = real.size(0)
    eps = torch.rand(bs, 1, device=real.device)
    eps = eps.expand_as(real)
    x_hat = eps * real + (1 - eps) * fake
    x_hat.requires_grad_(True)
    d_hat = D(x_hat)
    grad = torch.autograd.grad(
        outputs=d_hat.sum(), inputs=x_hat,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    gp = ((grad.norm(2, dim=1) - 1.0) ** 2).mean()
    return lambda_gp * gp
