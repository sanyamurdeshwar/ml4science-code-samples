from __future__ import annotations
import os, math, random, argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid, save_image

from datasets.mnist import get_mnist_loader
from datasets.generate_2d_toys import get_toy2d_loader

from .nets import GeneratorMNIST, CriticMNIST, Generator2D, Critic2D
from .metrics import sliced_wasserstein_2d, sliced_wasserstein_images, gradient_penalty

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_everything(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@dataclass
class TrainConfig:
    dataset: str = "swissroll"      # swissroll | checkerboard | gmm
    outdir: str = "outputs/wgan"
    epochs: int = 50
    batch_size: int = 256
    z_dim: int = 100
    lr: float = 5e-5
    n_critic: int = 5
    clip_value: float = 0.01
    seed: int = 1234
    # toy 2d
    n_samples: int = 8000
    num_mixture: int = 8
    radius: float = 8.0
    sigma: float = 1.0
    gp: bool = False
    lambda_gp: float = 10.0
    toy_hidden: int = 1024
    toy_depth: int = 5
    n_critic_toy: int = 10
    # mnist loader
    mnist_root: str = "datasets/mnist"
    num_workers: int = 4

def train_mnist(cfg: TrainConfig):
    outdir = Path(cfg.outdir) / "mnist"
    outdir.mkdir(parents=True, exist_ok=True)

    loader = get_mnist_loader(
        root=cfg.mnist_root,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        train=True,
        download=True,
    )

    G = GeneratorMNIST(cfg.z_dim).to(DEVICE)
    D = CriticMNIST().to(DEVICE)
    optD = torch.optim.RMSprop(D.parameters(), lr=cfg.lr)
    optG = torch.optim.RMSprop(G.parameters(), lr=cfg.lr)

    fixed_z = torch.randn(64, cfg.z_dim, device=DEVICE)

    for epoch in range(1, cfg.epochs + 1):
        running_d = running_g = 0.0
        for real, _ in loader:
            real = real.to(DEVICE)
            bs = real.size(0)

            for _ in range(cfg.n_critic):
                z = torch.randn(bs, cfg.z_dim, device=DEVICE)
                fake = G(z).detach()
                loss_d = -(D(real).mean() - D(fake).mean())
                optD.zero_grad(set_to_none=True)
                loss_d.backward()
                optD.step()
                for p in D.parameters():
                    p.data.clamp_(-cfg.clip_value, cfg.clip_value)

            z = torch.randn(bs, cfg.z_dim, device=DEVICE)
            fake = G(z)
            loss_g = -D(fake).mean()
            optG.zero_grad(set_to_none=True)
            loss_g.backward()
            optG.step()

            running_d += loss_d.item()
            running_g += loss_g.item()

        print(f"[MNIST] {epoch:03d}/{cfg.epochs} | D {running_d/len(loader):.4f} | G {running_g/len(loader):.4f}")

    G.eval()
    with torch.no_grad():
        final_fake = G(fixed_z).cpu()

    grid_fake = make_grid(final_fake, nrow=8, normalize=True, value_range=(-1, 1))
    save_image(grid_fake, outdir / "final_grid.png")

    real_batch, _ = next(iter(loader))
    real_batch = real_batch[:64]
    grid_real = make_grid(real_batch, nrow=8, normalize=True, value_range=(-1, 1))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    ax1.imshow(grid_real.permute(1, 2, 0).numpy()); ax1.set_title("Real"); ax1.axis("off")
    ax2.imshow(grid_fake.permute(1, 2, 0).numpy()); ax2.set_title("Generated"); ax2.axis("off")
    plt.tight_layout()
    plt.savefig(outdir / "real_vs_generated.png", dpi=200)
    plt.close()

    with torch.no_grad():
        z = torch.randn(1000, cfg.z_dim, device=DEVICE)
        gen_imgs = G(z)
    real_imgs = next(iter(get_mnist_loader(root=cfg.mnist_root, batch_size=1000, num_workers=0)))[0].to(DEVICE)
    swd = sliced_wasserstein_images(real_imgs, gen_imgs, device=DEVICE)
    print(f"Final MNIST SWD (pixel proxy): {swd:.6f}")

def train_toy2d(cfg: TrainConfig):
    name = cfg.dataset
    outdir = Path(cfg.outdir) / name.lower()
    outdir.mkdir(parents=True, exist_ok=True)

    # IMPORTANT: rename datasets/2d_toys -> datasets/toy2d so it’s importable.
    loader = get_toy2d_loader(
        name,
        batch_size=cfg.batch_size,
        n_samples=cfg.n_samples,
        seed=cfg.seed,
        num_mixture=cfg.num_mixture,
        radius=cfg.radius,
        sigma=cfg.sigma,
    )

    G = Generator2D(z_dim=cfg.z_dim, hidden=cfg.toy_hidden, depth=cfg.toy_depth).to(DEVICE)
    D = Critic2D(hidden=cfg.toy_hidden, depth=cfg.toy_depth).to(DEVICE)

    if cfg.gp:
        optD = torch.optim.Adam(D.parameters(), lr=1e-4, betas=(0.0, 0.9))
        optG = torch.optim.Adam(G.parameters(), lr=1e-4, betas=(0.0, 0.9))
        n_critic = cfg.n_critic_toy
    else:
        optD = torch.optim.RMSprop(D.parameters(), lr=cfg.lr)
        optG = torch.optim.RMSprop(G.parameters(), lr=cfg.lr)
        n_critic = cfg.n_critic

    for epoch in range(1, cfg.epochs + 1):
        running_d = running_g = 0.0
        for real in loader:
            real = real.to(DEVICE)
            bs = real.size(0)

            for _ in range(n_critic):
                z = torch.randn(bs, cfg.z_dim, device=DEVICE)
                fake = G(z).detach()
                loss_d = -(D(real).mean() - D(fake).mean())
                if cfg.gp:
                    loss_d = loss_d + gradient_penalty(D, real, fake, cfg.lambda_gp)

                optD.zero_grad(set_to_none=True)
                loss_d.backward()
                optD.step()

                if not cfg.gp:
                    for p in D.parameters():
                        p.data.clamp_(-cfg.clip_value, cfg.clip_value)

            z = torch.randn(bs, cfg.z_dim, device=DEVICE)
            fake = G(z)
            loss_g = -D(fake).mean()
            optG.zero_grad(set_to_none=True)
            loss_g.backward()
            optG.step()

            running_d += loss_d.item()
            running_g += loss_g.item()

        print(f"[{name}] {epoch:03d}/{cfg.epochs} | D {running_d/len(loader):.4f} | G {running_g/len(loader):.4f}")

    # viz + SWD
    real_all = []
    for batch in loader:
        real_all.append(batch)
        if len(real_all) * cfg.batch_size >= 10000:
            break
    real_all = torch.cat(real_all, dim=0).to(DEVICE)

    with torch.no_grad():
        z = torch.randn(real_all.size(0), cfg.z_dim, device=DEVICE)
        gen_all = G(z)

    rnp = real_all[:20000].detach().cpu().numpy()
    gnp = gen_all[:20000].detach().cpu().numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.scatter(rnp[:, 0], rnp[:, 1], s=5, alpha=0.5); ax1.set_title(f"Real ({name})"); ax1.grid(True, alpha=0.3)
    ax2.scatter(gnp[:, 0], gnp[:, 1], s=5, alpha=0.5); ax2.set_title("Generated"); ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / "real_vs_generated.png", dpi=200)
    plt.close()

    swd = sliced_wasserstein_2d(real_all[:10000].detach().cpu(), gen_all[:10000].detach().cpu(), device=DEVICE)
    print(f"Final {name} SWD: {swd:.6f}")

def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="swissroll",
                   choices=["swissroll", "checkerboard", "gmm"])
    p.add_argument("--outdir", type=str, default="outputs/wgan")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--z-dim", type=int, default=100)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--n-critic", type=int, default=5)
    p.add_argument("--clip-value", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=1234)

    # toy
    p.add_argument("--n-samples", type=int, default=8000)
    p.add_argument("--num-mixture", type=int, default=8)
    p.add_argument("--radius", type=float, default=8.0)
    p.add_argument("--sigma", type=float, default=1.0)
    p.add_argument("--gp", action="store_true")
    p.add_argument("--lambda-gp", type=float, default=10.0)
    p.add_argument("--toy-hidden", type=int, default=1024)
    p.add_argument("--toy-depth", type=int, default=5)
    p.add_argument("--n-critic-toy", type=int, default=10)

    p.add_argument("--num-workers", type=int, default=4)

    a = p.parse_args()
    return TrainConfig(
        dataset=a.dataset,
        outdir=a.outdir,
        epochs=a.epochs,
        batch_size=a.batch_size,
        z_dim=a.z_dim,
        lr=a.lr,
        n_critic=a.n_critic,
        clip_value=a.clip_value,
        seed=a.seed,
        n_samples=a.n_samples,
        num_mixture=a.num_mixture,
        radius=a.radius,
        sigma=a.sigma,
        gp=a.gp,
        lambda_gp=a.lambda_gp,
        toy_hidden=a.toy_hidden,
        toy_depth=a.toy_depth,
        n_critic_toy=a.n_critic_toy,
        mnist_root=a.mnist_root,
        num_workers=a.num_workers,
    )

def main():
    cfg = parse_args()
    seed_everything(cfg.seed)
    print("Device:", DEVICE)
    print("Config:", cfg)

    if cfg.dataset == "mnist":
        train_mnist(cfg)
    else:
        if not cfg.gp:
            print("Switching to WGAN-GP for toy2d (recommended). Use --gp to silence this.")
            cfg.gp = True
        train_toy2d(cfg)

if __name__ == "__main__":
    main()
