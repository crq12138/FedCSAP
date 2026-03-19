#!/usr/bin/env python3
"""Grid search runner for test_fl FL experiment attack reconstruction params."""

from __future__ import annotations

import argparse
import itertools
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run grid search on attack config params for test_fl.fl_system.main."
    )
    parser.add_argument("--python", default="python", help="Python executable to use.")
    parser.add_argument("--dataset", default="cifar10")
    parser.add_argument("--aggregation", default="fedavg")
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gaussian-noise-std", type=float, default=0.0)
    parser.add_argument("--num-clients", type=int, default=2)
    parser.add_argument("--attack", default="none")
    parser.add_argument("--mal-pcnt", type=float, default=0.0)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument(
        "--fixed-batch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to pass --fixed-batch to the FL command (default: enabled).",
    )
    parser.add_argument(
        "--output-root",
        default="grid_search_runs",
        help="Directory for generated configs, logs and summary files.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=20000,
        help="Fixed value for max_iterations in every run.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands/configs only; do not execute experiments.",
    )
    return parser.parse_args()


def value_grid() -> tuple[list[float], list[bool], list[float]]:
    total_variation_values = [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01]
    bn_stat_values = [True, False]
    bn_reg_scale_values = [0.0001, 0.0002, 0.0005, 0.001]
    return total_variation_values, bn_stat_values, bn_reg_scale_values


def write_attack_config(path: Path, *, tv: float, bn_stat: bool, bn_reg: float, max_iterations: int) -> None:
    config = {
        "signed": True,
        "boxed": True,
        "cost_fn": "sim",
        "lr": 0.1,
        "optim": "adam",
        "restarts": 1,
        "max_iterations": max_iterations,
        "total_variation": tv,
        "bn_stat": bn_stat,
        "bn_reg_scale": bn_reg,
        "z_norm": 0,
        "group_lazy": 0,
        "init": "randn",
        "filter": "none",
        "lr_decay": True,
        "scoring_choice": "loss",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")


def build_cmd(args: argparse.Namespace, attack_config_dir: Path) -> list[str]:
    cmd = [
        args.python,
        "-m",
        "test_fl.fl_system.main",
        "--dataset",
        args.dataset,
        "--aggregation",
        args.aggregation,
        "--local-epochs",
        str(args.local_epochs),
        "--batch-size",
        str(args.batch_size),
        "--gaussian-noise-std",
        str(args.gaussian_noise_std),
        "--num-clients",
        str(args.num_clients),
        f"--attack={args.attack}",
        f"--mal-pcnt={args.mal_pcnt}",
        "--rounds",
        str(args.rounds),
        "--attack-config-dir",
        str(attack_config_dir),
    ]
    if args.fixed_batch:
        cmd.append("--fixed-batch")
    return cmd


def iter_experiments() -> Iterable[tuple[float, bool, float]]:
    tv_values, bn_stat_values, bn_reg_values = value_grid()
    return itertools.product(tv_values, bn_stat_values, bn_reg_values)


def parse_best_loss(stdout: str) -> float | None:
    match = re.search(r"最优损失:\s*([0-9]+(?:\.[0-9]+)?)", stdout)
    if not match:
        return None
    return float(match.group(1))


def main() -> None:
    args = parse_args()
    run_stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.output_root) / f"run_{run_stamp}"
    config_dir = out_root / "attack_configs"
    logs_dir = out_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    best_record: dict | None = None

    for idx, (tv, bn_stat, bn_reg) in enumerate(iter_experiments(), start=1):
        write_attack_config(
            config_dir / f"bs{args.batch_size}.json",
            tv=tv,
            bn_stat=bn_stat,
            bn_reg=bn_reg,
            max_iterations=args.max_iterations,
        )

        cmd = build_cmd(args, attack_config_dir=config_dir)
        log_path = logs_dir / f"exp_{idx:03d}_tv{tv}_bn{int(bn_stat)}_bnreg{bn_reg}.log"

        print(f"[{idx:03d}] tv={tv}, bn_stat={bn_stat}, bn_reg_scale={bn_reg}, max_iter={args.max_iterations}")
        print(" ".join(cmd))

        if args.dry_run:
            results.append(
                {
                    "exp_id": idx,
                    "total_variation": tv,
                    "bn_stat": bn_stat,
                    "bn_reg_scale": bn_reg,
                    "max_iterations": args.max_iterations,
                    "status": "DRY_RUN",
                    "best_loss": None,
                    "log": str(log_path),
                }
            )
            continue

        proc = subprocess.run(cmd, text=True, capture_output=True)
        combined_out = f"# STDOUT\n{proc.stdout}\n\n# STDERR\n{proc.stderr}\n"
        log_path.write_text(combined_out, encoding="utf-8")
        best_loss = parse_best_loss(proc.stdout)

        record = {
            "exp_id": idx,
            "total_variation": tv,
            "bn_stat": bn_stat,
            "bn_reg_scale": bn_reg,
            "max_iterations": args.max_iterations,
            "status": "OK" if proc.returncode == 0 else f"FAILED({proc.returncode})",
            "best_loss": best_loss,
            "log": str(log_path),
        }
        results.append(record)

        if proc.returncode == 0 and best_loss is not None:
            if best_record is None or best_loss < best_record["best_loss"]:
                best_record = record

    summary = {
        "command_template": build_cmd(args, attack_config_dir=config_dir),
        "search_space": {
            "total_variation": value_grid()[0],
            "bn_stat": value_grid()[1],
            "bn_reg_scale": value_grid()[2],
            "max_iterations": args.max_iterations,
        },
        "results": results,
        "best": best_record,
    }

    summary_path = out_root / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nDone. Summary written to: {summary_path}")
    if best_record is None:
        print("No valid best result found (check logs).")
    else:
        print(
            "Best config => "
            f"tv={best_record['total_variation']}, "
            f"bn_stat={best_record['bn_stat']}, "
            f"bn_reg_scale={best_record['bn_reg_scale']}, "
            f"max_iterations={best_record['max_iterations']}, "
            f"best_loss={best_record['best_loss']}"
        )


if __name__ == "__main__":
    main()
