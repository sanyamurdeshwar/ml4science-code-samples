from __future__ import annotations
import random, argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from datasets.toy2d import get_toy2d_loader
from datasets.toy3d import get_toy3d_loader

from .nets import VAEConv, VAEMlp, VAEManifoldMlp
from .losses import kld_loss, sphere_rec_loss, hyper_rec_loss_x1x2x0, mse_rec_loss

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_everything(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@dataclass
class TrainConfig:
    dataset: str = "swissroll"  #swissroll | checkerboard | gmm | sphere | hyperboloid
    outdir: str = "outputs/geom_vae"
    epochs: int = 200
    batch_size: int = 512
    lr: float = 1e-3
    seed: int = 42
    zdim: int = 32
    # toy sampling
    n_samples: int = 8000
    num_mixture: int = 8 # for gmm / checkerboard
    radius: float = 8.0
    sigma: float = 1.0
    num_workers: int = 4

def _plot_points(real: np.ndarray, gen: np.ndarray, outpath: Path, title: str):
    fig = plt.figure(figsize=(6, 3))
    if real.shape[1] == 2:
        ax1 = fig.add_subplot(121); ax2 = fig.add_subplot(122)
        ax1.scatter(real[:,0], real[:,1], s=3, alpha=0.5); ax1.set_title("Real")
        ax2.scatter(gen[:,0], gen[:,1], s=3, alpha=0.7); ax2.set_title("VAE Gen")
        for ax in (ax1, ax2): ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
    else:
        ax1 = fig.add_subplot(121, projection="3d")
        ax2 = fig.add_subplot(122, projection="3d")
        ax1.scatter(real[:,0], real[:,1], real[:,2], s=1); ax1.set_title("Real")
        ax2.scatter(gen[:,0], gen[:,1], gen[:,2], s=1); ax2.set_title("VAE Gen")
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

def train(cfg: TrainConfig):
    seed_everything(cfg.seed)
    outdir = Path(cfg.outdir) / cfg.dataset.lower()
    outdir.mkdir(parents=True, exist_ok=True)

    ds = cfg.dataset.lower()


    # ---- 2D / 3D point clouds ----
    if ds in ["swissroll", "checkerboard", "gmm"]:
        loader = get_toy2d_loader(
            cfg.dataset,
            batch_size=cfg.batch_size,
            n_samples=cfg.n_samples,
            seed=cfg.seed,
            num_mixture=cfg.num_mixture,
            radius=cfg.radius,
            sigma=cfg.sigma,
        )
        indim = 2
        vae = VAEMlp(indim=indim, zdim=cfg.zdim).to(DEVICE)
        rec_fn = mse_rec_loss

    elif ds in ["sphere", "hyperboloid"]:
        loader = get_toy3d_loader(cfg.dataset, batch_size=cfg.batch_size, n_samples=cfg.n_samples, seed=cfg.seed)
        indim = 3
        vae = VAEManifoldMlp(indim=indim, zdim=cfg.zdim,
                             manifold_type=("Sphere" if ds == "sphere" else "Hyperboloid")).to(DEVICE)
        rec_fn = sphere_rec_loss if ds == "sphere" else hyper_rec_loss_x1x2x0

    else:
        raise ValueError(f"Unknown dataset: {cfg.dataset}")

    opt = torch.optim.Adam(vae.parameters(), lr=cfg.lr)

    for ep in range(1, cfg.epochs + 1):
        vae.train()
        kld_w = min(1.0, ep / 50.0) 
        meters = []
        for x in loader:
            x = x.to(DEVICE)
            xrec, mu, logvar = vae(x)
            rec = rec_fn(xrec, x)
            kld = kld_loss(mu, logvar)
            loss = rec + kld_w * kld

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            meters.append((loss.item(), rec.item(), kld.item()))

        if ep % 10 == 0 or ep == 1:
            L, R, K = np.mean(meters, axis=0)
            print(f"[{cfg.dataset}] ep {ep}/{cfg.epochs} loss {L:.4f} rec {R:.4f} kld {K:.4f}")

    # plot samples
    vae.eval()
    with torch.no_grad():
        z = torch.randn(5000, cfg.zdim, device=DEVICE)
        gen = vae.decode(z).cpu().numpy()

    # grab real samples
    real_batches = []
    for b in loader:
        real_batches.append(b.cpu())
        if len(real_batches) * cfg.batch_size >= 5000:
            break
    real = torch.cat(real_batches, dim=0)[:5000].numpy()

    _plot_points(real, gen[:5000], outdir / "real_vs_gen.png", title=cfg.dataset)

def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="swissroll",
                   choices=["swissroll", "checkerboard", "gmm", "sphere", "hyperboloid"])
    p.add_argument("--outdir", type=str, default="outputs/geom_vae")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--zdim", type=int, default=32)
    p.add_argument("--n-samples", type=int, default=8000)
    p.add_argument("--num-mixture", type=int, default=8)
    p.add_argument("--radius", type=float, default=8.0)
    p.add_argument("--sigma", type=float, default=1.0)
    p.add_argument("--num-workers", type=int, default=4)
    a = p.parse_args()
    return TrainConfig(**vars(a))

if __name__ == "__main__":
    train(parse_args())
