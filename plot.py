from __future__ import annotations
#!/usr/bin/env python3
"""绘图程序入口。

当前支持图：
1) FedCSAP 指定 run 的信誉 R 与委员会当选次数点位图。
"""
"""
python plot.py fedcsap_r_vs_committee \
  --details-csv result/fedcsap_last_r_and_committee_takeover_run_XXX_to_XXX_details.csv \
  --run-id 123 \
  --output result/fedcsap_run_123_r_vs_committee.png
"""


import argparse
import csv
from pathlib import Path
from typing import Iterable


CLIENT_COUNT = 25


class PlotDataError(Exception):
    """绘图数据异常。"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FedCSAP 绘图工具")
    subparsers = parser.add_subparsers(dest="plot_type", required=True)

    scatter_parser = subparsers.add_parser(
        "fedcsap_r_vs_committee",
        help="绘制指定 run 的 R 值 vs 委员会当选次数点位图",
    )
    scatter_parser.add_argument(
        "--details-csv",
        required=True,
        type=Path,
        help=(
            "由 scripts/read_runs_data.py 生成的 *_details.csv 文件路径，例如 "
            "result/fedcsap_last_r_and_committee_takeover_run_001_to_010_details.csv"
        ),
    )
    scatter_parser.add_argument(
        "--run-id",
        required=True,
        help="目标 run 编号。支持 7 或 007，程序会自动匹配。",
    )
    scatter_parser.add_argument(
        "--output",
        default="result/fedcsap_r_vs_committee.png",
        type=Path,
        help="图片输出路径（默认：result/fedcsap_r_vs_committee.png）",
    )
    scatter_parser.add_argument(
        "--title",
        default=None,
        help="图标题（可选，不传则自动生成）",
    )

    return parser.parse_args()


def normalize_run_id(run_id: str) -> tuple[int, str]:
    run_id_str = str(run_id).strip()
    if not run_id_str.isdigit():
        raise PlotDataError(f"run-id 必须为数字字符串，当前为: {run_id}")
    run_id_int = int(run_id_str)
    return run_id_int, run_id_str


def _client_columns(prefix: str) -> list[str]:
    return [f"client_{cid}_{prefix}" for cid in range(CLIENT_COUNT)]


def validate_details_columns(fieldnames: Iterable[str] | None) -> None:
    field_set = set(fieldnames or [])
    required_columns = {
        "run_id",
        *_client_columns("last_R"),
        *_client_columns("committee_elected_count"),
        *_client_columns("is_malicious"),
    }
    missing = sorted(required_columns - field_set)
    if missing:
        preview = ", ".join(missing[:5])
        more = "..." if len(missing) > 5 else ""
        raise PlotDataError(f"details CSV 缺失必要列: {preview}{more}")


def find_target_run_row(details_csv: Path, run_id: str) -> dict[str, str]:
    run_id_int, _ = normalize_run_id(run_id)

    with details_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        validate_details_columns(reader.fieldnames)

        for row in reader:
            raw = str(row.get("run_id", "")).strip()
            if not raw:
                continue
            if raw.isdigit() and int(raw) == run_id_int:
                return row

    raise PlotDataError(f"在 {details_csv} 中找不到 run_id={run_id} 对应的数据行。")


def build_scatter_points(run_row: dict[str, str]) -> tuple[list[float], list[float], list[bool], list[str]]:
    x_r_values: list[float] = []
    y_elected_counts: list[float] = []
    is_malicious_flags: list[bool] = []
    client_labels: list[str] = []

    for cid in range(CLIENT_COUNT):
        r_key = f"client_{cid}_last_R"
        elected_key = f"client_{cid}_committee_elected_count"
        malicious_key = f"client_{cid}_is_malicious"

        raw_r = str(run_row.get(r_key, "")).strip()
        raw_elected = str(run_row.get(elected_key, "")).strip()
        raw_malicious = str(run_row.get(malicious_key, "")).strip()

        if raw_r == "" or raw_elected == "":
            continue

        try:
            r_value = float(raw_r)
            elected_count = float(raw_elected)
        except ValueError as exc:
            raise PlotDataError(
                f"client_{cid} 的 R 或 committee_elected_count 不是数值: R={raw_r}, elected={raw_elected}"
            ) from exc

        malicious = False
        if raw_malicious != "":
            try:
                malicious = int(float(raw_malicious)) == 1
            except ValueError:
                malicious = False

        x_r_values.append(r_value)
        y_elected_counts.append(elected_count)
        is_malicious_flags.append(malicious)
        client_labels.append(f"client_{cid}")

    if not x_r_values:
        raise PlotDataError("目标 run 中没有可用于绘图的有效点位数据。")

    return x_r_values, y_elected_counts, is_malicious_flags, client_labels


def plot_fedcsap_r_vs_committee(details_csv: Path, run_id: str, output: Path, title: str | None = None) -> Path:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise PlotDataError("未安装 matplotlib，请先安装后再绘图（例如: pip install matplotlib）。") from exc

    if not details_csv.exists():
        raise PlotDataError(f"details CSV 不存在: {details_csv}")

    run_row = find_target_run_row(details_csv, run_id)
    x_r_values, y_elected_counts, is_malicious_flags, _ = build_scatter_points(run_row)

    normal_x = [x for x, m in zip(x_r_values, is_malicious_flags) if not m]
    normal_y = [y for y, m in zip(y_elected_counts, is_malicious_flags) if not m]
    malicious_x = [x for x, m in zip(x_r_values, is_malicious_flags) if m]
    malicious_y = [y for y, m in zip(y_elected_counts, is_malicious_flags) if m]

    fig, ax = plt.subplots(figsize=(10, 6))
    if normal_x:
        ax.scatter(normal_x, normal_y, s=70, alpha=0.85, label="Normal", marker="o")
    if malicious_x:
        ax.scatter(malicious_x, malicious_y, s=90, alpha=0.9, label="Malicious", marker="x", color="crimson")

    auto_title = f"FedCSAP Run {run_id}: R vs Committee Elected Count"
    ax.set_title(title or auto_title)
    ax.set_xlabel("Reputation R")
    ax.set_ylabel("Committee Elected Count")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def main() -> None:
    args = parse_args()

    if args.plot_type == "fedcsap_r_vs_committee":
        out = plot_fedcsap_r_vs_committee(
            details_csv=args.details_csv,
            run_id=args.run_id,
            output=args.output,
            title=args.title,
        )
        print(f"绘图完成，输出文件: {out}")
        return

    raise PlotDataError(f"不支持的 plot_type: {args.plot_type}")


if __name__ == "__main__":
    try:
        main()
    except PlotDataError as exc:
        raise SystemExit(f"[ERROR] {exc}") from exc
