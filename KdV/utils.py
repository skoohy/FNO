from typing import Tuple

import jax.numpy as jnp
import torch
import h5py
from torch.utils.data import DataLoader, Dataset


def to_coords(
    x: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    x_, t_ = torch.meshgrid(x, t)
    x_ = x_.T
    t_ = t_.T
    return torch.stack((x_, t_), -1)


class HDF5Dataset(Dataset):
    def __init__(
        self,
        path: str,
        mode: str,
        nt: int,
        nx: int,
        dtype=torch.float64,
        load_all: bool = False,
    ):
        super().__init__()
        f = h5py.File(path, "r")
        self.mode = mode
        self.dtype = dtype
        self.data = f[self.mode]
        self.dataset = f"pde_{nt}-{nx}"

        if load_all:
            data = {self.dataset: self.data[self.dataset][:]}
            f.close()
            self.data = data

    def __len__(
        self,
    ) -> int:
        return self.data[self.dataset].shape[0]

    def __getitem__(
        self,
        idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        u = self.data[self.dataset][idx]
        x = self.data["x"][idx]
        t = self.data["t"][idx]
        dx = self.data["dx"][idx]
        dt = self.data["dt"][idx]

        if self.mode == "train":
            X = to_coords(torch.tensor(x), torch.tensor(t))
            sol = (torch.tensor(u), X)
            u = sol[0]
            X = sol[1]
            dx = X[0, 1, 0] - X[0, 0, 0]
            dt = X[1, 0, 1] - X[0, 0, 1]
        else:
            u = torch.from_numpy(u)
            dx = torch.tensor([dx])
            dt = torch.tensor([dt])
        return u.float(), dx.float(), dt.float()


def create_dataloader(
    data_string: str,
    mode: str,
    nt: int,
    nx: int,
    batch_size: int,
) -> DataLoader:
    try:
        dataset = HDF5Dataset(data_string, mode, nt=nt, nx=nx)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    except:
        raise Exception("Datasets could not be loaded properly")
    return loader


def create_data(
    datapoints: torch.Tensor,
    start_time: list,
    time_future: int,
    time_history: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    data = torch.Tensor()
    labels = torch.Tensor()

    for dp, start in zip(datapoints, start_time):
        end_time = start + time_history
        d = dp[start:end_time]
        target_start_time = end_time
        target_end_time = target_start_time + time_future
        l = dp[target_start_time:target_end_time]

        data = torch.cat((data, d[None, :]), 0)
        labels = torch.cat((labels, l[None, :]), 0)
    return jnp.array(data), jnp.array(labels)
