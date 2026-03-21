from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch
import torchvision
import torchvision.transforms.functional as TF


def parse_args():
    p = argparse.ArgumentParser(
        description="Run 16 single-image reconstruction experiments (seed 0~15) and stitch results into a 4x4 panel."
    )
    p.add_argument("--dataset", choices=["cifar10", "mnist", "pathmnist"], default="cifar10")
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--rounds", type=int, default=1)
    p.add_argument("--local-epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-images", type=int, default=1)
    p.add_argument("--num-clients", type=int, default=5)
    p.add_argument("--attack", choices=["none", "sf"], default="none")
    p.add_argument("--mal-pcnt", type=float, default=0.0)
    p.add_argument("--attack-config-dir", default=None)
    p.add_argument("--noisy-std", type=float, default=0.001, help="Gaussian std for the noisy-gradient scenario.")
    p.add_argument("--fedcsap-alpha", type=float, default=0.5, help="Hybrid alpha for FEDCSAP scenario.")
    p.add_argument("--fedcsap-noise-std", type=float, default=0.0)
    p.add_argument("--output-dir", default="attack_results/compare16")
    p.add_argument("--python", default=sys.executable)
    return p.parse_args()


def _run(cmd: list[str]):
    print("[run]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _load_summary(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_resized_tensor(img_path: str, size: int = 96) -> torch.Tensor:
    img = torchvision.io.read_image(img_path).float() / 255.0
    if img.shape[0] == 1:
        img = img.repeat(3, 1, 1)
    if img.shape[0] > 3:
        img = img[:3]
    img = TF.resize(img, [size, size], antialias=True)
    return img


def save_4x4_grid(images: list[torch.Tensor], path: Path):
    if len(images) != 16:
        raise ValueError(f"Expected 16 images, got {len(images)}")
    grid = torchvision.utils.make_grid(images, nrow=4, padding=8, pad_value=0.96)
    torchvision.utils.save_image(grid, str(path))


def main():
    args = parse_args()
    if args.batch_size != 1 or args.num_images != 1:
        raise ValueError("This script requires --batch-size 1 and --num-images 1.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scenarios = {
        "grad": {
            "aggregation": "fedavg",
            "gaussian_noise_std": 0.0,
            "fedcsap_hybrid_alpha": 1.0,
        },
        "grad_noise": {
            "aggregation": "fedavg",
            "gaussian_noise_std": args.noisy_std,
            "fedcsap_hybrid_alpha": 1.0,
        },
        "fedcsap": {
            "aggregation": "fedcsap",
            "gaussian_noise_std": args.fedcsap_noise_std,
            "fedcsap_hybrid_alpha": args.fedcsap_alpha,
        },
    }

    all_records = []
    original_images: list[torch.Tensor] = []
    grad_images: list[torch.Tensor] = []
    grad_noise_images: list[torch.Tensor] = []
    fedcsap_images: list[torch.Tensor] = []

    for seed in range(16):
        per_seed = {}
        for name, sc in scenarios.items():
            cmd = [
                args.python,
                "-m",
                "test_fl.fl_system.main",
                "--dataset",
                args.dataset,
                "--data-dir",
                args.data_dir,
                "--aggregation",
                sc["aggregation"],
                "--local-epochs",
                str(args.local_epochs),
                "--batch-size",
                str(args.batch_size),
                "--num-images",
                str(args.num_images),
                "--gaussian-noise-std",
                str(sc["gaussian_noise_std"]),
                "--fedcsap-hybrid-alpha",
                str(sc["fedcsap_hybrid_alpha"]),
                "--num-clients",
                str(args.num_clients),
                "--attack",
                args.attack,
                "--mal-pcnt",
                str(args.mal_pcnt),
                "--rounds",
                str(args.rounds),
                "--seed",
                str(seed),
            ]
            if args.attack_config_dir:
                cmd.extend(["--attack-config-dir", args.attack_config_dir])
            _run(cmd)

            summary_path = Path(
                "attack_results/"
                f"attack_summary_seed{seed}_agg{sc['aggregation']}_"
                f"alpha{sc['fedcsap_hybrid_alpha']}_noise{sc['gaussian_noise_std']}_bs{args.num_images}.json"
            )
            if not summary_path.exists():
                raise FileNotFoundError(f"Expected summary file not found: {summary_path}")
            per_seed[name] = _load_summary(summary_path)

        original_images.append(_load_resized_tensor(per_seed["grad"]["target_image_path"]))
        grad_images.append(_load_resized_tensor(per_seed["grad"]["reconstruction_path"]))
        grad_noise_images.append(_load_resized_tensor(per_seed["grad_noise"]["reconstruction_path"]))
        fedcsap_images.append(_load_resized_tensor(per_seed["fedcsap"]["reconstruction_path"]))
        all_records.append(
            {
                "seed": seed,
                "loss_grad": per_seed["grad"]["reconstruction_loss_opt"],
                "loss_grad_noise": per_seed["grad_noise"]["reconstruction_loss_opt"],
                "loss_fedcsap": per_seed["fedcsap"]["reconstruction_loss_opt"],
                "paths": per_seed,
            }
        )

    original_grid_path = output_dir / "compare16_original_4x4.png"
    grad_grid_path = output_dir / "compare16_grad_4x4.png"
    grad_noise_grid_path = output_dir / "compare16_grad_noise_4x4.png"
    fedcsap_grid_path = output_dir / "compare16_fedcsap_4x4.png"
    save_4x4_grid(original_images, original_grid_path)
    save_4x4_grid(grad_images, grad_grid_path)
    save_4x4_grid(grad_noise_images, grad_noise_grid_path)
    save_4x4_grid(fedcsap_images, fedcsap_grid_path)

    json_path = output_dir / "compare16_losses.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(all_records, f, ensure_ascii=False, indent=2)

    print(f"[done] 原图 4x4 已保存: {original_grid_path}")
    print(f"[done] 原始梯度重建 4x4 已保存: {grad_grid_path}")
    print(f"[done] 噪声梯度重建 4x4 已保存: {grad_noise_grid_path}")
    print(f"[done] FEDCSAP 重建 4x4 已保存: {fedcsap_grid_path}")
    print(f"[done] loss 汇总已保存: {json_path}")


if __name__ == "__main__":
    main()
