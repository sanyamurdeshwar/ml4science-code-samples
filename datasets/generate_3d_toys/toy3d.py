from __future__ import annotations
import math
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class SphereCheckerBoardFull(Dataset):
    def __init__(self, n_samples: int = 8000, seed: int = 42,
                 tiles_theta: int = 12, tiles_mu: int = 12):
        rng = np.random.default_rng(seed)

        theta_edges = np.linspace(0.0, 2.0 * np.pi, tiles_theta + 1)
        mu_edges = np.linspace(-1.0, 1.0, tiles_mu + 1)

        selected = [(i, j) for i in range(tiles_theta) for j in range(tiles_mu) if (i + j) % 2 == 0]
        n_tiles = len(selected)
        base = n_samples // n_tiles
        extra = n_samples % n_tiles

        pts = []
        for idx, (i, j) in enumerate(selected):
            k = base + (1 if idx < extra else 0)
            t0, t1 = theta_edges[i], theta_edges[i + 1]
            m0, m1 = mu_edges[j], mu_edges[j + 1]

            thetas = rng.uniform(t0, t1, size=k)
            mus = rng.uniform(m0, m1, size=k)  # z = mu
            phis = np.arccos(np.clip(mus, -1.0, 1.0))

            x = np.sin(phis) * np.cos(thetas)
            y = np.sin(phis) * np.sin(thetas)
            z = np.cos(phis)
            pts.append(np.stack([x, y, z], axis=1))

        all_pts = np.concatenate(pts, axis=0)
        order = rng.permutation(all_pts.shape[0])
        self.coords_3d = torch.from_numpy(all_pts[order]).float()

    def __len__(self) -> int: return self.coords_3d.shape[0]
    def __getitem__(self, idx: int) -> torch.Tensor: return self.coords_3d[idx]

class HyperboloidCheckerBoardFull(Dataset):
    def __init__(self, n_samples: int = 8000, seed: int = 42,
                 tiles_theta: int = 24, tiles_m: int = 12, R: float = 3.2):
        rng = np.random.default_rng(seed)
        theta_edges = np.linspace(0.0, 2.0 * np.pi, tiles_theta + 1)
        m_edges = np.linspace(0.0, 1.0, tiles_m + 1)
        C = math.cosh(float(R)) - 1.0

        selected = [(i, j) for i in range(tiles_theta) for j in range(tiles_m) if (i + j) % 2 == 0]
        n_tiles = len(selected)
        base = n_samples // n_tiles
        extra = n_samples % n_tiles

        pts = []
        for idx, (i, j) in enumerate(selected):
            k = base + (1 if idx < extra else 0)
            t0, t1 = theta_edges[i], theta_edges[i + 1]
            m0, m1 = m_edges[j], m_edges[j + 1]

            thetas = rng.uniform(t0, t1, size=k)
            ms = rng.uniform(m0, m1, size=k)

            cosh_r = 1.0 + C * ms
            rs = np.arccosh(np.maximum(cosh_r, 1.0))
            sinh_r = np.sinh(rs)

            x1 = sinh_r * np.cos(thetas)
            x2 = sinh_r * np.sin(thetas)
            x0 = cosh_r
            pts.append(np.stack([x1, x2, x0], axis=1))

        all_pts = np.concatenate(pts, axis=0)
        order = rng.permutation(all_pts.shape[0])
        self.coords_3d = torch.from_numpy(all_pts[order]).float()

    def __len__(self) -> int: return self.coords_3d.shape[0]
    def __getitem__(self, idx: int) -> torch.Tensor: return self.coords_3d[idx]

def get_toy3d_loader(
    name: str,
    *,
    batch_size: int,
    n_samples: int,
    seed: int,
) -> DataLoader:
    key = name.lower()
    if key in ["sphere", "spherical", "s2"]:
        ds = SphereCheckerBoardFull(n_samples=n_samples, seed=seed)
    elif key in ["hyperboloid", "hyperbolic", "h2"]:
        ds = HyperboloidCheckerBoardFull(n_samples=n_samples, seed=seed)
    else:
        raise ValueError(f"Unknown toy3d dataset: {name}")

    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
