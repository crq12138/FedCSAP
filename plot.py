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
PLOT_OUTPUT_DIR = Path("plot_result")
DEFAULT_RUN_MAPPING = {
    "CIFAR": {
        373: "FedAvg",
        374: "Median",
        375: "KRUM",
        376: "FoolsGold",
        377: "FLTrust",
        378: "AFA",
        379: "FedCSAP",
    },
    "PATHMNIST": {
        380: "FedAvg",
        381: "Median",
        382: "KRUM",
        383: "FoolsGold",
        384: "FLTrust",
        385: "AFA",
        386: "FedCSAP",
    },
    "MNIST": {
        387: "FedAvg",
        388: "Median",
        389: "KRUM",
        390: "FoolsGold",
        391: "FLTrust",
        392: "AFA",
        393: "FedCSAP",
    },
}
DATASET_ROUND_LIMITS = {"CIFAR": 200, "PATHMNIST": 150, "MNIST": 100}
SCHEME_NAME_CANONICAL = {
    "fedavg": "FedAvg",
    "median": "Median",
    "krum": "KRUM",
    "foolsgold": "FoolsGold",
    "fltrust": "FLTrust",
    "afa": "AFA",
    "fedcsap": "FedCSAP",
}
SCHEME_MARKERS = {
    "FedAvg": "o",
    "Median": "s",
    "KRUM": "^",
    "FoolsGold": "D",
    "FLTrust": "v",
    "AFA": "P",
    "FedCSAP": "*",
}
SCHEME_COLORS = {
    "FedCSAP": "#d62728",
}

# ========== 统一图形元素尺寸配置区域（便于调试）==========
PLOT_STYLE = {
    "figure_size": (7.2, 4.2),  # IEEE 双栏常见宽高比例
    "normal_marker_size": 90,
    "malicious_marker_size": 90,
    "label_font_size": 20,
    "tick_font_size": 18,
    "legend_font_size": 20,
    "annotation_font_size": 16,
    "annotation_offset": (3, 2),  # (x, y) 偏移，单位 points
    "grid_linewidth": 0.7,
    "spine_linewidth": 0.8,
    "dpi": 300,
}


class PlotDataError(Exception):
    """绘图数据异常。"""


def canonicalize_scheme_name(raw_name: str) -> str:
    normalized = "".join(str(raw_name).strip().split()).lower()
    return SCHEME_NAME_CANONICAL.get(normalized, str(raw_name).strip())


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
        default="plot_result/fedcsap_r_vs_committee",
        type=Path,
        help=(
            "输出文件名前缀（可带/不带后缀）。"
            "程序会固定保存到 plot_result 目录下，并同时导出 .png 与 .eps。"
        ),
    )
    scatter_parser.add_argument(
        "--title",
        default=None,
        help="图标题（可选，不传则自动生成）",
    )
    curve_parser = subparsers.add_parser(
        "compare_training_curves",
        help="绘制对比实验训练曲线（ACC/F1）",
    )
    curve_parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("runs"),
        help="runs 根目录，目录结构示例: runs/run_373/global_metrics.csv",
    )
    curve_parser.add_argument(
        "--output-dir",
        type=Path,
        default=PLOT_OUTPUT_DIR / "compare_training_curves",
        help="输出目录（自动生成 6 张图）",
    )
    curve_parser.add_argument(
        "--run-map-csv",
        type=Path,
        default=None,
        help="可选：run_id, dataset, scheme 三列映射文件；不传则使用内置映射。",
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
        client_labels.append(f"{cid}")

    if not x_r_values:
        raise PlotDataError("目标 run 中没有可用于绘图的有效点位数据。")

    return x_r_values, y_elected_counts, is_malicious_flags, client_labels


def plot_fedcsap_r_vs_committee(details_csv: Path, run_id: str, output: Path, title: str | None = None) -> Path:
    try:
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
        from adjustText import adjust_text  # <--- 新增：引入排斥力算法库
    except ModuleNotFoundError as exc:
        raise PlotDataError("未安装必要的绘图库，请先执行: pip install matplotlib adjustText") from exc # <--- 修改：提示中加入 adjustText

    if not details_csv.exists():
        raise PlotDataError(f"details CSV 不存在: {details_csv}")
    try:
        simhei_font_path = font_manager.findfont("SimHei", fallback_to_default=False)
    except ValueError as exc:
        raise PlotDataError(
            "未找到 SimHei 字体。请先在系统中安装 SimHei（如 simhei.ttf），再执行绘图。"
        ) from exc
    simhei_font = font_manager.FontProperties(fname=simhei_font_path)

    run_row = find_target_run_row(details_csv, run_id)
    x_r_values, y_elected_counts, is_malicious_flags, client_labels = build_scatter_points(run_row)

    normal_x = [x for x, m in zip(x_r_values, is_malicious_flags) if not m]
    normal_y = [y for y, m in zip(y_elected_counts, is_malicious_flags) if not m]
    malicious_x = [x for x, m in zip(x_r_values, is_malicious_flags) if m]
    malicious_y = [y for y, m in zip(y_elected_counts, is_malicious_flags) if m]

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["SimHei", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "axes.linewidth": PLOT_STYLE["spine_linewidth"],
            "axes.grid": True,
            "grid.linestyle": "--",
            "grid.alpha": 0.35,
            "grid.linewidth": PLOT_STYLE["grid_linewidth"],
            "savefig.bbox": "tight",
        }
    )

    fig, ax = plt.subplots(figsize=PLOT_STYLE["figure_size"])
    if normal_x:
        ax.scatter(
            normal_x,
            normal_y,
            s=PLOT_STYLE["normal_marker_size"],
            alpha=0.88,
            label="正常参与方",
            marker="o",
            color="#1f77b4",
            edgecolors="white",
            linewidths=0.5,
        )
    if malicious_x:
        ax.scatter(
            malicious_x,
            malicious_y,
            s=PLOT_STYLE["malicious_marker_size"],
            alpha=0.95,
            label="恶意参与方",
            marker="X",
            color="#d62728",
            edgecolors="white",
            linewidths=0.5,
        )

    # 需求：图中不显示标题（title 参数仅为兼容保留）
    _ = title
    ax.set_xlabel("信誉值 R", fontsize=PLOT_STYLE["label_font_size"], fontproperties=simhei_font)
    ax.set_ylabel("委员会当选次数", fontsize=PLOT_STYLE["label_font_size"], fontproperties=simhei_font)
    # 实例化一个专用于刻度的 FontProperties，并赋予指定字号
    tick_font = font_manager.FontProperties(
        fname=simhei_font_path, 
        size=PLOT_STYLE["tick_font_size"]
    )
    # 取消原本的 ax.tick_params，直接通过 set_fontproperties 一步到位
    for tick in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
        tick.set_fontproperties(tick_font)

    # 标注每个点对应 client_id
    # for x, y, cid, is_malicious in zip(x_r_values, y_elected_counts, client_labels, is_malicious_flags):
    #     ax.annotate(
    #         cid,
    #         (x, y),
    #         textcoords="offset points",
    #         xytext=PLOT_STYLE["annotation_offset"],
    #         fontsize=PLOT_STYLE["annotation_font_size"],
    #         color="#d62728" if is_malicious else "#1f77b4",
    #         alpha=0.95,
    #     )
    texts = []
    for x, y, cid, is_malicious in zip(x_r_values, y_elected_counts, client_labels, is_malicious_flags):
        txt = ax.text(
            x, 
            y, 
            cid,
            fontsize=PLOT_STYLE["annotation_font_size"],
            color="#d62728" if is_malicious else "#1f77b4",
            alpha=0.95,
            fontproperties=simhei_font,
        )
        texts.append(txt)

    # 运用斥力算法自动计算非重叠位置
    adjust_text(
        texts,
        x=x_r_values,
        y=y_elected_counts,
        ax=ax,
        expand_points=(1.5, 1.5),  # 调整文本与数据点之间的排斥力阈值
        expand_text=(1.2, 1.2),    # 调整文本与文本之间的排斥力阈值
        # arrowprops=dict(arrowstyle="-", color="gray", lw=0.6, alpha=0.7) # 偏移后自动生成指向原数据点的细灰线
        arrowprops=None,
    )

    # 横坐标范围必须包含 1.0
    x_min_data = min(x_r_values)
    x_max_data = max(x_r_values)
    margin = max((x_max_data - x_min_data) * 0.08, 0.02)
    x_low = min(x_min_data - margin, 1.0)
    x_high = max(x_max_data + margin, 1.0)
    if x_low == x_high:
        x_high = x_low + 1.0
    ax.set_xlim(x_low, x_high)

    # 保证刻度里出现 1.0
    current_ticks = list(ax.get_xticks())
    if not any(abs(t - 1.0) < 1e-9 for t in current_ticks):
        current_ticks.append(1.0)
        current_ticks = sorted(set(round(t, 6) for t in current_ticks))
        ax.set_xticks(current_ticks)

    # 实例化一个专用于图例的 FontProperties，并赋予指定字号
    legend_font = font_manager.FontProperties(
        fname=simhei_font_path, 
        size=PLOT_STYLE["legend_font_size"]
    )
    # 移除被忽略的 fontsize 参数，只保留 prop
    ax.legend(frameon=True, prop=legend_font)

    output_stem = output.stem if output.suffix else output.name
    png_path = PLOT_OUTPUT_DIR / f"{output_stem}.png"
    eps_path = PLOT_OUTPUT_DIR / f"{output_stem}.eps"
    PLOT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(png_path, dpi=PLOT_STYLE["dpi"])
    fig.savefig(eps_path, format="eps")
    plt.close(fig)
    print(f"绘图完成，输出文件: {png_path}")
    print(f"绘图完成，输出文件: {eps_path}")
    return png_path


def _load_run_mapping(run_map_csv: Path | None) -> dict[str, dict[int, str]]:
    if run_map_csv is None:
        return DEFAULT_RUN_MAPPING
    if not run_map_csv.exists():
        raise PlotDataError(f"run 映射文件不存在: {run_map_csv}")
    mapping: dict[str, dict[int, str]] = {}
    with run_map_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"run_id", "dataset", "scheme"}
        fields = set(reader.fieldnames or [])
        if not required.issubset(fields):
            raise PlotDataError("run 映射 CSV 必须包含列: run_id,dataset,scheme")
        for row in reader:
            run_id = int(str(row["run_id"]).strip())
            dataset = str(row["dataset"]).strip().upper()
            scheme = canonicalize_scheme_name(str(row["scheme"]))
            if dataset not in DATASET_ROUND_LIMITS:
                continue
            mapping.setdefault(dataset, {})[run_id] = scheme
    return mapping


def _load_global_metrics(
    metrics_csv: Path,
    round_limit: int,
    sample_every: int = 5,
) -> tuple[list[int], list[float], list[float]]:
    with metrics_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"epoch", "global_acc", "global_macro_f1"}
        fields = set(reader.fieldnames or [])
        if not required.issubset(fields):
            raise PlotDataError(f"{metrics_csv} 缺失必要列: epoch,global_acc,global_macro_f1")
        rows = list(reader)[:round_limit]
    if sample_every > 1:
        rows = [r for r in rows if int(float(str(r["epoch"]).strip())) % sample_every == 0]
    if not rows:
        raise PlotDataError(f"{metrics_csv} 无可用数据。")
    epochs = [int(float(str(r["epoch"]).strip())) for r in rows]
    acc = [float(str(r["global_acc"]).strip()) for r in rows]
    f1 = [float(str(r["global_macro_f1"]).strip()) for r in rows]
    return epochs, acc, f1


def plot_compare_training_curves(
    runs_root: Path,
    output_dir: Path,
    run_map_csv: Path | None = None,
) -> list[Path]:
    try:
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
    except ModuleNotFoundError as exc:
        raise PlotDataError("未安装 matplotlib，请先执行: pip install matplotlib") from exc

    if not runs_root.exists():
        raise PlotDataError(f"runs 根目录不存在: {runs_root}")

    try:
        simhei_font_path = font_manager.findfont("SimHei", fallback_to_default=False)
    except ValueError as exc:
        raise PlotDataError("未找到 SimHei 字体，请先安装 SimHei（simhei.ttf）。") from exc
    simhei_font = font_manager.FontProperties(fname=simhei_font_path)

    run_mapping = _load_run_mapping(run_map_csv)
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["SimHei", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.linestyle": "--",
            "grid.alpha": 0.35,
            "grid.linewidth": 0.7,
            "legend.frameon": True,
            "savefig.bbox": "tight",
        }
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for dataset in ("CIFAR", "PATHMNIST", "MNIST"):
        dataset_runs = run_mapping.get(dataset, {})
        if not dataset_runs:
            continue
        round_limit = DATASET_ROUND_LIMITS[dataset]
        curves: dict[str, dict[str, list[float] | list[int]]] = {}
        for run_id, scheme in sorted(dataset_runs.items()):
            csv_path = runs_root / f"run_{run_id}" / "global_metrics.csv"
            if not csv_path.exists():
                raise PlotDataError(f"缺少文件: {csv_path}")
            epochs, acc, f1 = _load_global_metrics(csv_path, round_limit, sample_every=5)
            curves[scheme] = {"epoch": epochs, "acc": acc, "f1": f1}
        for metric_key, metric_cn in (("acc", "ACC"), ("f1", "F1")):
            fig, ax = plt.subplots(figsize=(7.2, 4.2))
            for scheme, vals in curves.items():
                is_fedcsap = scheme == "FedCSAP"
                ax.plot(
                    vals["epoch"],
                    vals[metric_key],
                    linewidth=2.8 if is_fedcsap else 2.0,
                    label=scheme,
                    marker=SCHEME_MARKERS.get(scheme, "o"),
                    markersize=8 if is_fedcsap else 6,
                    color=SCHEME_COLORS.get(scheme),
                )
            ax.set_xlabel("轮次", fontsize=16, fontproperties=simhei_font)
            ax.set_ylabel(metric_cn, fontsize=16, fontproperties=simhei_font)
            ax.set_xlim(1, round_limit)
            for tick in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
                tick.set_fontproperties(simhei_font)
                tick.set_fontsize(12)
            legend_font = font_manager.FontProperties(fname=simhei_font_path, size=12)
            ax.legend(prop=legend_font, ncol=2)
            fig.tight_layout()
            out_png = output_dir / f"{dataset}_{metric_key}_curves.png"
            out_eps = output_dir / f"{dataset}_{metric_key}_curves.eps"
            fig.savefig(out_png, dpi=300)
            fig.savefig(out_eps, format="eps")
            plt.close(fig)
            saved.extend([out_png, out_eps])
            print(f"绘图完成: {out_png}")
            print(f"绘图完成: {out_eps}")
    if not saved:
        raise PlotDataError("未生成任何图，请检查 run 映射配置。")
    return saved


def main() -> None:
    args = parse_args()

    if args.plot_type == "fedcsap_r_vs_committee":
        out = plot_fedcsap_r_vs_committee(
            details_csv=args.details_csv,
            run_id=args.run_id,
            output=args.output,
            title=args.title,
        )
        print(f"主输出文件: {out}")
        return
    if args.plot_type == "compare_training_curves":
        outs = plot_compare_training_curves(
            runs_root=args.runs_root,
            output_dir=args.output_dir,
            run_map_csv=args.run_map_csv,
        )
        print(f"主输出文件: {outs[0]}")
        return

    raise PlotDataError(f"不支持的 plot_type: {args.plot_type}")


if __name__ == "__main__":
    try:
        main()
    except PlotDataError as exc:
        raise SystemExit(f"[ERROR] {exc}") from exc
