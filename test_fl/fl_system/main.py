from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch

from .client import FLClient
from .config import FLConfig
from .datasets import build_client_loaders, build_test_loader, load_dataset
from .model import build_model
from .server import FLServer


def parse_args():
    p = argparse.ArgumentParser(description="Standalone Federated Learning System (from scratch)")
    p.add_argument("--dataset", choices=["cifar10", "mnist", "pathmnist"], default="cifar10")
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--rounds", type=int, default=200)
    p.add_argument("--num-clients", type=int, default=20)
    p.add_argument("--dirichlet-alpha", type=float, default=0.9)
    p.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=64)
    p.add_argument(
        "--num-images",
        "--num_images",
        dest="num_images",
        type=int,
        default=None,
        help="Number of images used for gradient reconstruction attack. Defaults to --batch-size.",
    )
    p.add_argument("--lr", type=float, default=0.1)
    # p.add_argument("--eta", type=float, default=0.1)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--local-epochs", type=int, default=2)
    p.add_argument("--aggregation", choices=["fedavg", "fedcsap"], default="fedavg")
    p.add_argument("--fedcsap-hybrid-alpha", type=float, default=1.0,
                   help="Hybrid update strength for FEDCSAP in (0,1]. 1.0 means pure aggregated update.")
    p.add_argument("--gaussian-noise-std", type=float, default=0.0,
                   help="Std of Gaussian noise added to aggregated model parameters.")
    p.add_argument("--attack", choices=["none", "sf"], default="sf")
    p.add_argument("--fixed-batch", action="store_true",
                   help="Use the same fixed training batch for local training and reconstruction attack.")
    p.add_argument("--mal-pcnt", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    p.add_argument("--max-train-samples-per-client", type=int, default=None)
    p.add_argument("--max-test-samples", type=int, default=None)
    p.add_argument("--attack-config-dir", default=None,
                   help="Directory that contains attack config JSON files named as bs{batch_size}.json.")
    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    args = parse_args()
    if args.num_images is not None and args.num_images <= 0:
        raise ValueError("--num-images must be a positive integer.")
    cfg = FLConfig(
        dataset=args.dataset,
        data_dir=args.data_dir,
        rounds=args.rounds,
        num_clients=args.num_clients,
        dirichlet_alpha=args.dirichlet_alpha,
        batch_size=args.batch_size,
        lr=args.lr,
        # eta=args.eta,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        local_epochs=args.local_epochs,
        aggregation=args.aggregation,
        attack=args.attack,
        mal_pcnt=args.mal_pcnt,
        num_images=args.num_images,
        fedcsap_hybrid_alpha=args.fedcsap_hybrid_alpha,
        gaussian_noise_std=args.gaussian_noise_std,
        seed=args.seed,
        fixed_batch=args.fixed_batch,
        device=resolve_device(args.device),
        max_train_samples_per_client=args.max_train_samples_per_client,
        max_test_samples=args.max_test_samples,
    )
    if args.attack_config_dir is not None:
        cfg.attack_config_dir = Path(args.attack_config_dir)

    if cfg.num_clients > 20:
        raise ValueError("根据需求，联邦学习客户端数量不能超过20个。")

    set_seed(cfg.seed)

    train_ds, test_ds, labels, num_classes, in_channels = load_dataset(cfg.dataset, cfg.data_dir)
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

    clients = [FLClient(i, loader, cfg.device, fixed_batch=cfg.fixed_batch) for i, loader in enumerate(client_loaders)]
    model = build_model(cfg.dataset, num_classes=num_classes, in_channels=in_channels)

    print(
        f"Start FL: dataset={cfg.dataset}, clients={cfg.num_clients}, rounds={cfg.rounds}, "
        f"attack={cfg.attack}, mal_pcnt={cfg.mal_pcnt}, aggregation={cfg.aggregation}, "
        f"hybrid_alpha={cfg.fedcsap_hybrid_alpha}, noise_std={cfg.gaussian_noise_std}, device={cfg.device}"
    )

    server = FLServer(model=model, clients=clients, test_loader=test_loader, cfg=cfg)
    server.train()


if __name__ == "__main__":
    main()
