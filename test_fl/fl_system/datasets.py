from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def _make_indices_by_dirichlet(labels: np.ndarray, num_clients: int, alpha: float, seed: int) -> list[list[int]]:
    rng = np.random.default_rng(seed)
    num_classes = int(labels.max() + 1)
    client_indices = [[] for _ in range(num_clients)]

    for c in range(num_classes):
        class_idx = np.where(labels == c)[0]
        rng.shuffle(class_idx)
        proportions = rng.dirichlet([alpha] * num_clients)
        cuts = (np.cumsum(proportions) * len(class_idx)).astype(int)[:-1]
        splits = np.split(class_idx, cuts)
        for cid, idxs in enumerate(splits):
            client_indices[cid].extend(idxs.tolist())

    for cid in range(num_clients):
        rng.shuffle(client_indices[cid])
    return client_indices


def _limited_subset(dataset, max_samples: int | None):
    if max_samples is None or max_samples >= len(dataset):
        return dataset
    return Subset(dataset, list(range(max_samples)))


def load_dataset(name: str, data_dir: str):
    name = name.lower()
    data_root = Path(data_dir).expanduser().resolve()
    data_root.mkdir(parents=True, exist_ok=True)

    if name == "cifar10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])
        train_ds = datasets.CIFAR10(root=str(data_root), train=True, download=True, transform=transform)
        test_ds = datasets.CIFAR10(root=str(data_root), train=False, download=True, transform=transform)
        labels = np.array(train_ds.targets)
        num_classes = 10
        return train_ds, test_ds, labels, num_classes, 3

    if name == "mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        train_ds = datasets.MNIST(root=str(data_root), train=True, download=True, transform=transform)
        test_ds = datasets.MNIST(root=str(data_root), train=False, download=True, transform=transform)
        labels = np.array(train_ds.targets)
        num_classes = 10
        return train_ds, test_ds, labels, num_classes, 1

    if name == "pathmnist":
        try:
            import medmnist
            from medmnist import PathMNIST
        except ImportError as exc:
            raise RuntimeError("PATHMNIST 需要安装 medmnist: pip install medmnist") from exc

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        train_ds = PathMNIST(split="train", root=str(data_root), download=True, transform=transform)
        test_ds = PathMNIST(split="test", root=str(data_root), download=True, transform=transform)
        labels = np.array(train_ds.labels).reshape(-1)
        num_classes = len(medmnist.INFO["pathmnist"]["label"])
        return train_ds, test_ds, labels, num_classes, 3

    raise ValueError(f"Unsupported dataset: {name}")


def build_client_loaders(
    train_ds,
    labels: np.ndarray,
    num_clients: int,
    dirichlet_alpha: float,
    seed: int,
    batch_size: int,
    max_train_samples_per_client: int | None = None,
):
    client_indices = _make_indices_by_dirichlet(labels, num_clients, dirichlet_alpha, seed)
    loaders = []
    for idxs in client_indices:
        subset = Subset(train_ds, idxs)
        subset = _limited_subset(subset, max_train_samples_per_client)
        loaders.append(DataLoader(subset, batch_size=batch_size, shuffle=True, drop_last=False))
    return loaders


def build_test_loader(test_ds, batch_size: int, max_test_samples: int | None = None):
    test_ds = _limited_subset(test_ds, max_test_samples)
    return DataLoader(test_ds, batch_size=batch_size, shuffle=False)
