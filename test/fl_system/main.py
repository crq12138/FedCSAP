from __future__ import annotations

import argparse
import random

import numpy as np
import torch

from .client import FLClient
from .config import FLConfig
from .datasets import build_client_loaders, build_test_loader, load_dataset
from .model import SmallCNN
from .server import FLServer


def parse_args():
    p = argparse.ArgumentParser(description="Standalone Federated Learning System (from scratch)")
    p.add_argument("--dataset", choices=["cifar10", "pathmnist"], default="cifar10")
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--rounds", type=int, default=200)
    p.add_argument("--num-clients", type=int, default=20)
    p.add_argument("--dirichlet-alpha", type=float, default=0.9)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--local-epochs", type=int, default=1)
    p.add_argument("--aggregation", default="fedavg")
    p.add_argument("--attack", choices=["none", "sf"], default="none")
    p.add_argument("--mal-pcnt", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--max-train-samples-per-client", type=int, default=None)
    p.add_argument("--max-test-samples", type=int, default=None)
    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    cfg = FLConfig(
        dataset=args.dataset,
        data_dir=args.data_dir,
        rounds=args.rounds,
        num_clients=args.num_clients,
        dirichlet_alpha=args.dirichlet_alpha,
        batch_size=args.batch_size,
        lr=args.lr,
        local_epochs=args.local_epochs,
        aggregation=args.aggregation,
        attack=args.attack,
        mal_pcnt=args.mal_pcnt,
        seed=args.seed,
        device=args.device,
        max_train_samples_per_client=args.max_train_samples_per_client,
        max_test_samples=args.max_test_samples,
    )

    if cfg.num_clients != 20:
        raise ValueError("根据需求，联邦学习客户端数量必须保持在20个。")

    set_seed(cfg.seed)

    train_ds, test_ds, labels, num_classes = load_dataset(cfg.dataset, cfg.data_dir)
    client_loaders = build_client_loaders(
        train_ds,
        labels,
        num_clients=cfg.num_clients,
        dirichlet_alpha=cfg.dirichlet_alpha,
        seed=cfg.seed,
        batch_size=cfg.batch_size,
        max_train_samples_per_client=cfg.max_train_samples_per_client,
    )
    test_loader = build_test_loader(test_ds, cfg.batch_size, cfg.max_test_samples)

    clients = [FLClient(i, loader, cfg.device) for i, loader in enumerate(client_loaders)]
    model = SmallCNN(num_classes=num_classes)

    print(
        f"Start FL: dataset={cfg.dataset}, clients={cfg.num_clients}, rounds={cfg.rounds}, "
        f"attack={cfg.attack}, mal_pcnt={cfg.mal_pcnt}, aggregation={cfg.aggregation}"
    )

    server = FLServer(model=model, clients=clients, test_loader=test_loader, cfg=cfg)
    server.train()


if __name__ == "__main__":
    main()
