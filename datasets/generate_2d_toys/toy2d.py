from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import sklearn.datasets

class SwissRoll(Dataset):
    def __init__(self, n_samples: int = 8000, seed: int = 42) -> None:
        np.random.seed(seed)
        data = sklearn.datasets.make_swiss_roll(n_samples=n_samples, noise=1.0)[0]
        data = data.astype("float32")[:, [0, 2]]
        data /= 4.0
        data = torch.from_numpy(data).float()

        r = 4.5
        data1 = data.clone() + torch.tensor([-r, -r])
        data2 = data.clone() + torch.tensor([-r,  r])
        data3 = data.clone() + torch.tensor([ r, -r])
        data4 = data.clone() + torch.tensor([ r,  r])
        self.x = torch.cat([data, data1, data2, data3, data4], dim=0)

    def __len__(self) -> int: return self.x.shape[0]
    def __getitem__(self, idx: int) -> torch.Tensor: return self.x[idx]

class CheckerBoard(Dataset):
    def __init__(self, n_samples: int = 8000, seed: int = 42) -> None:
        np.random.seed(seed)
        x1 = np.random.rand(n_samples) * 4 - 2
        x2_ = np.random.rand(n_samples) - np.random.randint(0, 2, n_samples) * 2
        x2 = x2_ + (np.floor(x1) % 2)
        self.x = torch.from_numpy(np.concatenate([x1[:, None], x2[:, None]], 1) * 2).float()

    def __len__(self) -> int: return self.x.shape[0]
    def __getitem__(self, idx: int) -> torch.Tensor: return self.x[idx]

class NGaussianMixtures(Dataset):
    def __init__(
        self,
        num_mixture: int = 8,
        n_samples: int = 40000,
        n_dimensional: int = 2,
        radius: float = 8.0,
        sigma: float = 1.0,
        seed: int = 42,
    ) -> None:
        np.random.seed(seed)
        mix_probs = torch.tensor([1/num_mixture] * num_mixture)
        mix_idx = torch.multinomial(mix_probs, n_samples, replacement=True)

        thetas = np.linspace(0, 2 * np.pi, num_mixture, endpoint=False)
        xs = radius * np.sin(thetas, dtype=np.float32)
        ys = radius * np.cos(thetas, dtype=np.float32)
        centers = torch.tensor(np.vstack([xs, ys]).T)[mix_idx]  # [N,2]

        stds = torch.ones(n_samples, n_dimensional) * sigma
        self.x = torch.randn_like(centers) * stds + centers

    def __len__(self) -> int: return self.x.shape[0]
    def __getitem__(self, idx: int) -> torch.Tensor: return self.x[idx]

def make_toy2d_dataset(
    name: str,
    *,
    n_samples: int,
    seed: int,
    num_mixture: int,
    radius: float,
    sigma: float,
) -> Dataset:
    key = name.lower()
    if key in ["swissroll", "swiss_roll", "swiss-roll"]:
        return SwissRoll(n_samples=n_samples, seed=seed)
    if key in ["checkerboard", "checker_board", "checker"]:
        return CheckerBoard(n_samples=n_samples, seed=seed)
    if key in ["gmm", "mixture", "ngaussians", "n_gaussians"]:
        return NGaussianMixtures(
            num_mixture=num_mixture,
            n_samples=max(n_samples, 40000),  # keep your original behavior if you want
            n_dimensional=2,
            radius=radius,
            sigma=sigma,
            seed=seed,
        )
    raise ValueError(f"Unknown toy dataset: {name}")

def get_toy2d_loader(
    name: str,
    *,
    batch_size: int,
    n_samples: int,
    seed: int,
    num_mixture: int = 8,
    radius: float = 8.0,
    sigma: float = 1.0,
) -> DataLoader:
    ds = make_toy2d_dataset(
        name,
        n_samples=n_samples,
        seed=seed,
        num_mixture=num_mixture,
        radius=radius,
        sigma=sigma,
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
