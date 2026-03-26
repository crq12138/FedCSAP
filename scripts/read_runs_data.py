#!/usr/bin/env python3
"""读取 runs 目录指定范围 run_* 子目录中的结果数据并汇总。"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from statistics import mean
from typing import Callable


MetricHandler = Callable[[Path, range, int], dict]


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
        required=True,
        help="起始 run 编号，例如 225",
    )
    parser.add_argument(
        "--end-run",
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


def parse_run_bounds(start_run: str, end_run: str) -> tuple[int, int, int]:
    start_run_str = start_run.strip()
    end_run_str = end_run.strip()
    if not start_run_str.isdigit() or not end_run_str.isdigit():
        raise DataReadError("run 编号必须为数字字符串，例如 009 或 225。")

    start_run_int = int(start_run_str)
    end_run_int = int(end_run_str)
    width = max(len(start_run_str), len(end_run_str))

    return start_run_int, end_run_int, width


def resolve_run_csv_path(runs_dir: Path, run_id: int, run_id_width: int, filename: str) -> Path:
    padded = runs_dir / f"run_{run_id:0{run_id_width}d}" / filename
    if padded.exists():
        return padded

    plain = runs_dir / f"run_{run_id}" / filename
    return plain


def read_poisontest_accuracy_avg(runs_dir: Path, run_ids: range, run_id_width: int) -> dict:
    accuracies: list[float] = []
    per_run_averages: dict[int, float] = {}
    missing_files: list[str] = []
    empty_rows: list[str] = []
    missing_columns: list[str] = []
    invalid_values: list[str] = []
    no_valid_accuracy_runs: list[str] = []

    for run_id in run_ids:
        csv_path = resolve_run_csv_path(
            runs_dir,
            run_id,
            run_id_width,
            "posiontest_result.csv",
        )
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


def read_global_macro_f1_max(runs_dir: Path, run_ids: range, run_id_width: int) -> dict:
    max_values: dict[int, float] = {}
    missing_files: list[str] = []
    empty_rows: list[str] = []
    missing_columns: list[str] = []
    invalid_values: list[str] = []
    no_valid_f1_runs: list[str] = []

    for run_id in run_ids:
        csv_path = resolve_run_csv_path(
            runs_dir,
            run_id,
            run_id_width,
            "global_metrics.csv",
        )
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
                run_f1_values.append(float(raw_f1))
            except ValueError:
                invalid_values.append(
                    f"{csv_path} [row {row_idx}] (invalid global_macro_f1={raw_f1})"
                )

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

def read_fedcsap_last_r_and_committee_takeover(runs_dir: Path, run_ids: range, run_id_width: int) -> dict:
    client_ids = list(range(25))
    per_run_last_r: dict[int, dict[int, float]] = {}
    per_run_committee_takeover_count: dict[int, int] = {}
    per_run_committee_elected_count: dict[int, dict[int, int]] = {}
    per_run_is_malicious: dict[int, dict[int, float]] = {}
    missing_files: list[str] = []
    empty_rows: list[str] = []
    missing_columns: list[str] = []
    invalid_values: list[str] = []

    for run_id in run_ids:
        client_csv = resolve_run_csv_path(
            runs_dir,
            run_id,
            run_id_width,
            "fedcsap_client_metrics.csv",
        )
        round_csv = resolve_run_csv_path(
            runs_dir,
            run_id,
            run_id_width,
            "fedcsap_round_metrics.csv",
        )

        if not client_csv.exists():
            missing_files.append(str(client_csv))
            continue
        if not round_csv.exists():
            missing_files.append(str(round_csv))
            continue

        with client_csv.open("r", encoding="utf-8", newline="") as f:
            client_reader = csv.DictReader(f)
            client_rows = list(client_reader)

        if not client_rows:
            empty_rows.append(str(client_csv))
            continue

        required_client_columns = {"client_id", "R", "is_malicious"}
        client_fieldnames = set(client_reader.fieldnames or [])
        if not required_client_columns.issubset(client_fieldnames):
            missing_columns.append(str(client_csv))
            continue

        last_r_by_client: dict[int, float] = {}
        appearance_count_by_client: dict[int, int] = {cid: 0 for cid in client_ids}
        is_malicious_observed_values: dict[int, set[int]] = {cid: set() for cid in client_ids}

        for row_idx, row in enumerate(client_rows, start=1):
            raw_client_id = row.get("client_id")
            raw_is_malicious = row.get("is_malicious")

            if raw_client_id is None or str(raw_client_id).strip() == "":
                invalid_values.append(f"{client_csv} [row {row_idx}] (empty client_id)")
                continue

            try:
                client_id = int(raw_client_id)
            except ValueError:
                invalid_values.append(
                    f"{client_csv} [row {row_idx}] (invalid client_id={raw_client_id})"
                )
                continue

            if client_id not in client_ids:
                continue

            appearance_count_by_client[client_id] += 1

            if raw_is_malicious is None or str(raw_is_malicious).strip() == "":
                invalid_values.append(
                    f"{client_csv} [row {row_idx}] (empty is_malicious for client_id={client_id})"
                )
            else:
                try:
                    is_malicious_val = int(float(raw_is_malicious))
                    if is_malicious_val not in (0, 1):
                        raise ValueError
                    is_malicious_observed_values[client_id].add(is_malicious_val)
                except ValueError:
                    invalid_values.append(
                        f"{client_csv} [row {row_idx}] (invalid is_malicious={raw_is_malicious} for client_id={client_id})"
                    )

        for row_idx, row in enumerate(reversed(client_rows), start=1):
            raw_client_id = row.get("client_id")
            raw_r = row.get("R")
            if raw_client_id is None or str(raw_client_id).strip() == "":
                invalid_values.append(f"{client_csv} [from_end_row {row_idx}] (empty client_id)")
                continue
            if raw_r is None or str(raw_r).strip() == "":
                invalid_values.append(
                    f"{client_csv} [from_end_row {row_idx}] (empty R for client_id={raw_client_id})"
                )
                continue

            try:
                client_id = int(raw_client_id)
                r_value = float(raw_r)
            except ValueError:
                invalid_values.append(
                    f"{client_csv} [from_end_row {row_idx}] (invalid client_id={raw_client_id} or R={raw_r})"
                )
                continue

            if client_id not in client_ids:
                continue

            if client_id not in last_r_by_client:
                last_r_by_client[client_id] = r_value
                if len(last_r_by_client) == len(client_ids):
                    break

        is_malicious_by_client: dict[int, float] = {}
        for cid in client_ids:
            observed = is_malicious_observed_values[cid]
            if not observed:
                is_malicious_by_client[cid] = math.nan
                continue
            if len(observed) > 1:
                invalid_values.append(
                    f"{client_csv} (conflicting is_malicious values for client_id={cid}: {sorted(observed)})"
                )
            is_malicious_by_client[cid] = float(max(observed))

        with round_csv.open("r", encoding="utf-8", newline="") as f:
            round_reader = csv.DictReader(f)
            round_rows = list(round_reader)

        if not round_rows:
            empty_rows.append(str(round_csv))
            continue

        if "committee_takeover" not in round_reader.fieldnames if round_reader.fieldnames else True:
            missing_columns.append(str(round_csv))
            continue

        committee_takeover_count = 0
        for row_idx, row in enumerate(round_rows, start=1):
            raw_takeover = row.get("committee_takeover")
            if raw_takeover is None or str(raw_takeover).strip() == "":
                invalid_values.append(
                    f"{round_csv} [row {row_idx}] (empty committee_takeover)"
                )
                continue
            try:
                takeover_value = float(raw_takeover)
            except ValueError:
                invalid_values.append(
                    f"{round_csv} [row {row_idx}] (invalid committee_takeover={raw_takeover})"
                )
                continue
            if takeover_value != 0:
                committee_takeover_count += 1

        total_round_count = len(round_rows)
        committee_elected_count_by_client = {
            cid: max(total_round_count - appearance_count_by_client[cid], 0)
            for cid in client_ids
        }

        per_run_last_r[run_id] = {cid: last_r_by_client.get(cid, math.nan) for cid in client_ids}
        per_run_committee_takeover_count[run_id] = committee_takeover_count
        per_run_committee_elected_count[run_id] = committee_elected_count_by_client
        per_run_is_malicious[run_id] = is_malicious_by_client

    if not per_run_last_r:
        error_messages = ["未读取到可用 FedCSAP client/round metrics 数据。"]
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
                f"缺失列文件数: {len(missing_columns)}（示例: {missing_columns[0]}）"
            )
        if invalid_values:
            error_messages.append(
                f"无效数值数: {len(invalid_values)}（示例: {invalid_values[0]}）"
            )
        raise DataReadError("\n".join(error_messages))

    return {
        "task": "fedcsap_last_r_and_committee_takeover",
        "total_runs": len(run_ids),
        "valid_run_count": len(per_run_last_r),
        "per_run_last_r": per_run_last_r,
        "per_run_committee_takeover_count": per_run_committee_takeover_count,
        "per_run_committee_elected_count": per_run_committee_elected_count,
        "per_run_is_malicious": per_run_is_malicious,
        "missing_files": missing_files,
        "empty_rows": empty_rows,
        "missing_columns": missing_columns,
        "invalid_values": invalid_values,
    }


def get_task_handlers() -> dict[str, MetricHandler]:
    """任务注册表：为后续功能扩展预留接口。"""
    return {
        "poisontest_accuracy_avg": read_poisontest_accuracy_avg,
        "global_macro_f1_max": read_global_macro_f1_max,
        "fedcsap_last_r_and_committee_takeover": read_fedcsap_last_r_and_committee_takeover,
    }


def write_summary(output_dir: Path, start_run: str, end_run: str, summary: dict) -> tuple[Path, Path | None]:
    output_dir.mkdir(parents=True, exist_ok=True)
    base_filename = f"{summary['task']}_run_{start_run}_to_{end_run}"
    out_file = output_dir / f"{base_filename}.csv"
    extra_out_file = None
    
    run_label_width = max(len(start_run), len(end_run))

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
            run_label = f"{run_id:0{run_label_width}d}"
            rows.append(
                (
                    f"run_{run_label}_accuracy_avg",
                    f"{summary['per_run_averages'][run_id]:.10f}",
                )
            )
    elif summary["task"] == "global_macro_f1_max":
        rows.insert(3, ("max_global_macro_f1_avg", f"{summary['max_global_macro_f1_avg']:.10f}"))
        rows.append(("no_valid_f1_runs", len(summary["no_valid_f1_runs"])))

        # 新增：额外写入纯数值文件的逻辑
        extra_out_file = output_dir / f"{base_filename}_raw_values.txt"
        with extra_out_file.open("w", encoding="utf-8") as ef:
            for run_id in sorted(summary["per_run_max_global_macro_f1"]):
                run_label = f"{run_id:0{run_label_width}d}"
                f1_value = summary['per_run_max_global_macro_f1'][run_id]

                # 写入汇总表
                rows.append(
                    (
                        f"run_{run_label}_global_macro_f1_max",
                        f"{f1_value:.10f}",
                    )
                )
                # 写入纯数值表（附带换行符）
                ef.write(f"{f1_value:.10f}\n")
    elif summary["task"] == "fedcsap_last_r_and_committee_takeover":
        rows.insert(3, ("client_count", 25))

        extra_out_file = output_dir / f"{base_filename}_details.csv"
        with extra_out_file.open("w", encoding="utf-8", newline="") as ef:
            writer = csv.writer(ef)
            writer.writerow([
                "run_id",
                *[f"client_{cid}_last_R" for cid in range(25)],
                *[f"client_{cid}_committee_elected_count" for cid in range(25)],
                *[f"client_{cid}_is_malicious" for cid in range(25)],
                "committee_takeover_count",
            ])
            for run_id in sorted(summary["per_run_last_r"]):
                run_label = f"{run_id:0{run_label_width}d}"
                takeover_count = summary["per_run_committee_takeover_count"].get(run_id, 0)
                rows.append((f"run_{run_label}_committee_takeover_count", takeover_count))

                per_client = summary["per_run_last_r"][run_id]
                per_client_committee_elected_count = summary["per_run_committee_elected_count"][run_id]
                per_client_is_malicious = summary["per_run_is_malicious"][run_id]
                writer.writerow([
                    run_label,
                    *[
                        "" if math.isnan(per_client[cid]) else f"{per_client[cid]:.10f}"
                        for cid in range(25)
                    ],
                    *[
                        per_client_committee_elected_count[cid]
                        for cid in range(25)
                    ],
                    *[
                        "" if math.isnan(per_client_is_malicious[cid]) else int(per_client_is_malicious[cid])
                        for cid in range(25)
                    ],
                    takeover_count,
                ])

    with out_file.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerows(rows)

    return out_file, extra_out_file


def main() -> None:
    args = parse_args()

    runs_dir = Path(args.runs_dir)
    if not runs_dir.exists() or not runs_dir.is_dir():
        raise DataReadError(f"runs 目录不存在或不是文件夹: {runs_dir}")

    start_run_int, end_run_int, run_id_width = parse_run_bounds(args.start_run, args.end_run)
    run_ids = parse_run_range(start_run_int, end_run_int)
    handlers = get_task_handlers()

    if args.task not in handlers:
        available = ", ".join(sorted(handlers))
        raise DataReadError(f"未知 task: {args.task}。可选 task: {available}")

    summary = handlers[args.task](runs_dir, run_ids, run_id_width)
    out_file, extra_out_file = write_summary(Path(args.output_dir), args.start_run, args.end_run, summary)

    print("读取完成。")
    print(f"任务: {summary['task']}")
    print(f"run 范围: run_{args.start_run} 到 run_{args.end_run}")
    print(f"有效 run 数量: {summary['valid_run_count']}")
    if summary["task"] == "poisontest_accuracy_avg":
        print(f"可用 accuracy 数量: {summary['valid_accuracy_count']}")
        print(f"平均 accuracy: {summary['accuracy_avg']:.10f}")
    elif summary["task"] == "global_macro_f1_max":
        print(f"平均最大 global_macro_f1: {summary['max_global_macro_f1_avg']:.10f}")
    elif summary["task"] == "fedcsap_last_r_and_committee_takeover":
        print("已提取每个 run 的 25 个 client 的最后一次 R 及 committee_takeover 次数。")
    print(f"输出汇总文件: {out_file}")
    
    # 终端回显纯数值文件的生成情况
    if extra_out_file:
        print(f"输出纯数值文件: {extra_out_file}")


if __name__ == "__main__":
    try:
        main()
    except DataReadError as exc:
        raise SystemExit(f"[ERROR] {exc}") from exc
