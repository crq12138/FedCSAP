#!/usr/bin/env python3
"""读取 runs 目录指定范围 run_* 子目录中的结果数据并汇总。"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from statistics import mean
from typing import Callable


MetricHandler = Callable[[Path, range], dict]


class DataReadError(Exception):
    """数据读取相关错误。"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="读取 runs 中指定 run 范围的数据并输出汇总结果。"
    )
    parser.add_argument(
        "--runs-dir",
        required=True,
        help="runs 根目录路径，例如 ./runs",
    )
    parser.add_argument(
        "--start-run",
        type=int,
        required=True,
        help="起始 run 编号，例如 225",
    )
    parser.add_argument(
        "--end-run",
        type=int,
        required=True,
        help="结束 run 编号，例如 300",
    )
    parser.add_argument(
        "--output-dir",
        default="result",
        help="结果输出目录（默认：result）",
    )
    parser.add_argument(
        "--task",
        default="poisontest_accuracy_avg",
        help="要执行的任务名（默认：poisontest_accuracy_avg）",
    )
    return parser.parse_args()


def parse_run_range(start_run: int, end_run: int) -> range:
    if start_run <= 0 or end_run <= 0:
        raise DataReadError("run 编号必须为正整数。")
    if start_run > end_run:
        raise DataReadError("起始 run 编号不能大于结束 run 编号。")
    return range(start_run, end_run + 1)


def read_poisontest_accuracy_avg(runs_dir: Path, run_ids: range) -> dict:
    accuracies: list[float] = []
    per_run_averages: dict[int, float] = {}
    missing_files: list[str] = []
    empty_rows: list[str] = []
    missing_columns: list[str] = []
    invalid_values: list[str] = []
    no_valid_accuracy_runs: list[str] = []

    for run_id in run_ids:
        csv_path = runs_dir / f"run_{run_id}" / "posiontest_result.csv"
        if not csv_path.exists():
            missing_files.append(str(csv_path))
            continue

        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if not rows:
            empty_rows.append(str(csv_path))
            continue

        if "accuracy" not in reader.fieldnames if reader.fieldnames else True:
            missing_columns.append(str(csv_path))
            continue
        
        run_accuracies: list[float] = []

        for row_idx, row in enumerate(rows, start=1):
            raw_accuracy = row.get("accuracy")
            if raw_accuracy is None or str(raw_accuracy).strip() == "":
                invalid_values.append(f"{csv_path} [row {row_idx}] (empty accuracy)")
                continue
            try:
                accuracy = float(raw_accuracy)
                run_accuracies.append(accuracy)
                accuracies.append(accuracy)
            except ValueError:
                invalid_values.append(
                    f"{csv_path} [row {row_idx}] (invalid accuracy={raw_accuracy})"
                )
        if run_accuracies:
            per_run_averages[run_id] = mean(run_accuracies)
        else:
            no_valid_accuracy_runs.append(str(csv_path))

    if not per_run_averages:
        error_messages = ["未读取到可用 poisontest accuracy 数据，无法计算每个 run 的均值。"]
        if missing_files:
            error_messages.append(
                f"缺失文件数: {len(missing_files)}（示例: {missing_files[0]}）"
            )
        if empty_rows:
            error_messages.append(
                f"空文件数: {len(empty_rows)}（示例: {empty_rows[0]}）"
            )
        if missing_columns:
            error_messages.append(
                f"缺失 accuracy 列文件数: {len(missing_columns)}（示例: {missing_columns[0]}）"
            )
        if invalid_values:
            error_messages.append(
                f"无效 accuracy 数值数: {len(invalid_values)}（示例: {invalid_values[0]}）"
            )
        if no_valid_accuracy_runs:
            error_messages.append(
                f"无有效 accuracy 的 run 文件数: {len(no_valid_accuracy_runs)}（示例: {no_valid_accuracy_runs[0]}）"
            )
        raise DataReadError("\n".join(error_messages))

    return {
        "task": "poisontest_accuracy_avg",
        "total_runs": len(run_ids),
        "valid_run_count": len(per_run_averages),
        "valid_accuracy_count": len(accuracies),
        "accuracy_avg": mean(per_run_averages.values()),
        "per_run_averages": per_run_averages,
        "missing_files": missing_files,
        "empty_rows": empty_rows,
        "missing_columns": missing_columns,
        "invalid_values": invalid_values,
        "no_valid_accuracy_runs": no_valid_accuracy_runs,
    }


def read_global_macro_f1_max(runs_dir: Path, run_ids: range) -> dict:
    max_values: dict[int, float] = {}
    missing_files: list[str] = []
    empty_rows: list[str] = []
    missing_columns: list[str] = []
    invalid_values: list[str] = []
    no_valid_f1_runs: list[str] = []

    for run_id in run_ids:
        csv_path = runs_dir / f"run_{run_id}" / "global_metrics.csv"
        if not csv_path.exists():
            missing_files.append(str(csv_path))
            continue

        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if not rows:
            empty_rows.append(str(csv_path))
            continue

        if "global_macro_f1" not in reader.fieldnames if reader.fieldnames else True:
            missing_columns.append(str(csv_path))
            continue

        run_f1_values: list[float] = []
        for row_idx, row in enumerate(rows, start=1):
            raw_f1 = row.get("global_macro_f1")
            if raw_f1 is None or str(raw_f1).strip() == "":
                invalid_values.append(f"{csv_path} [row {row_idx}] (empty global_macro_f1)")
                continue
            try:
                f1_value = float(raw_f1)
            except ValueError:
                invalid_values.append(
                    f"{csv_path} [row {row_idx}] (invalid global_macro_f1={raw_f1})"
                )
                continue

            if not math.isfinite(f1_value):
                invalid_values.append(
                    f"{csv_path} [row {row_idx}] (non-finite global_macro_f1={raw_f1})"
                )
                continue

            run_f1_values.append(f1_value)

        if run_f1_values:
            max_values[run_id] = max(run_f1_values)
        else:
            no_valid_f1_runs.append(str(csv_path))

    if not max_values:
        error_messages = ["未读取到可用 global_macro_f1 数据，无法提取每个 run 的最大值。"]
        if missing_files:
            error_messages.append(
                f"缺失文件数: {len(missing_files)}（示例: {missing_files[0]}）"
            )
        if empty_rows:
            error_messages.append(
                f"空文件数: {len(empty_rows)}（示例: {empty_rows[0]}）"
            )
        if missing_columns:
            error_messages.append(
                f"缺失 global_macro_f1 列文件数: {len(missing_columns)}（示例: {missing_columns[0]}）"
            )
        if invalid_values:
            error_messages.append(
                f"无效 global_macro_f1 数值数: {len(invalid_values)}（示例: {invalid_values[0]}）"
            )
        if no_valid_f1_runs:
            error_messages.append(
                f"无有效 global_macro_f1 的 run 文件数: {len(no_valid_f1_runs)}（示例: {no_valid_f1_runs[0]}）"
            )
        raise DataReadError("\n".join(error_messages))

    return {
        "task": "global_macro_f1_max",
        "total_runs": len(run_ids),
        "valid_run_count": len(max_values),
        "max_global_macro_f1_avg": mean(max_values.values()),
        "per_run_max_global_macro_f1": max_values,
        "missing_files": missing_files,
        "empty_rows": empty_rows,
        "missing_columns": missing_columns,
        "invalid_values": invalid_values,
        "no_valid_f1_runs": no_valid_f1_runs,
    }


def get_task_handlers() -> dict[str, MetricHandler]:
    """任务注册表：为后续功能扩展预留接口。"""
    return {
        "poisontest_accuracy_avg": read_poisontest_accuracy_avg,
        "global_macro_f1_max": read_global_macro_f1_max,
    }


def write_summary(output_dir: Path, start_run: int, end_run: int, summary: dict) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"{summary['task']}_run_{start_run}_to_{end_run}.csv"

    rows = [
        ("task", summary["task"]),
        ("total_runs", summary["total_runs"]),
        ("valid_run_count", summary["valid_run_count"]),
        ("missing_files", len(summary["missing_files"])),
        ("empty_rows", len(summary["empty_rows"])),
        ("missing_columns", len(summary["missing_columns"])),
        ("invalid_values", len(summary["invalid_values"])),
    ]

    if summary["task"] == "poisontest_accuracy_avg":
        rows.insert(3, ("valid_accuracy_count", summary["valid_accuracy_count"]))
        rows.insert(4, ("accuracy_avg", f"{summary['accuracy_avg']:.10f}"))
        rows.append(("no_valid_accuracy_runs", len(summary["no_valid_accuracy_runs"])))
        for run_id in sorted(summary["per_run_averages"]):
            rows.append(
                (
                    f"run_{run_id}_accuracy_avg",
                    f"{summary['per_run_averages'][run_id]:.10f}",
                )
            )
    elif summary["task"] == "global_macro_f1_max":
        rows.insert(3, ("max_global_macro_f1_avg", f"{summary['max_global_macro_f1_avg']:.10f}"))
        rows.append(("no_valid_f1_runs", len(summary["no_valid_f1_runs"])))
        for run_id in sorted(summary["per_run_max_global_macro_f1"]):
            rows.append(
                (
                    f"run_{run_id}_global_macro_f1_max",
                    f"{summary['per_run_max_global_macro_f1'][run_id]:.10f}",
                )
            )

    with out_file.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerows(rows)

    return out_file


def main() -> None:
    args = parse_args()

    runs_dir = Path(args.runs_dir)
    if not runs_dir.exists() or not runs_dir.is_dir():
        raise DataReadError(f"runs 目录不存在或不是文件夹: {runs_dir}")

    run_ids = parse_run_range(args.start_run, args.end_run)
    handlers = get_task_handlers()

    if args.task not in handlers:
        available = ", ".join(sorted(handlers))
        raise DataReadError(f"未知 task: {args.task}。可选 task: {available}")

    summary = handlers[args.task](runs_dir, run_ids)
    out_file = write_summary(Path(args.output_dir), args.start_run, args.end_run, summary)

    print("读取完成。")
    print(f"任务: {summary['task']}")
    print(f"run 范围: run_{args.start_run} 到 run_{args.end_run}")
    print(f"有效 run 数量: {summary['valid_run_count']}")
    if summary["task"] == "poisontest_accuracy_avg":
        print(f"可用 accuracy 数量: {summary['valid_accuracy_count']}")
        print(f"平均 accuracy: {summary['accuracy_avg']:.10f}")
    elif summary["task"] == "global_macro_f1_max":
        print(f"平均最大 global_macro_f1: {summary['max_global_macro_f1_avg']:.10f}")
    print(f"输出文件: {out_file}")


if __name__ == "__main__":
    try:
        main()
    except DataReadError as exc:
        raise SystemExit(f"[ERROR] {exc}") from exc
