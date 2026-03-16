#!/usr/bin/env bash
set -euo pipefail

# 实验一：bottom q 最佳值实验
# 从配置表读取 run 配置，并按 run 区间启动（行为与 exp_04 系列一致）。
#
# 用法：
#   bash scripts/exp_01.sh
#   bash scripts/exp_01.sh 1 4
#   bash scripts/exp_01.sh run_001 run_004
#
# 可选环境变量：
#   MAX_PARALLEL=4
#   CONFIG_FILE=scripts/configs/exp_01_runs.csv
#   DRY_RUN=1   # 只打印命令，不实际启动

MAX_PARALLEL=${MAX_PARALLEL:-4}
CONFIG_FILE=${CONFIG_FILE:-scripts/configs/exp_01_runs.csv}
DRY_RUN=${DRY_RUN:-0}

if ! [[ "${MAX_PARALLEL}" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_PARALLEL must be a positive integer, got: ${MAX_PARALLEL}" >&2
  exit 1
fi

if [[ ! -f "${CONFIG_FILE}" ]]; then
  echo "Config file not found: ${CONFIG_FILE}" >&2
  exit 1
fi

normalize_run_id() {
  local input="$1"
  if [[ "${input}" =~ ^run_([0-9]+)$ ]]; then
    echo $((10#${BASH_REMATCH[1]}))
  elif [[ "${input}" =~ ^([0-9]+)$ ]]; then
    echo $((10#${BASH_REMATCH[1]}))
  else
    echo "Invalid run id: ${input}. Use 1 or run_001 format." >&2
    exit 1
  fi
}

START_INPUT=${1:-run_001}
END_INPUT=${2:-run_308}
START_ID=$(normalize_run_id "${START_INPUT}")
END_ID=$(normalize_run_id "${END_INPUT}")

if (( START_ID > END_ID )); then
  echo "Start run must be <= end run. Got: ${START_INPUT} .. ${END_INPUT}" >&2
  exit 1
fi

start_run() {
  local run_tag="$1"
  local type="$2"
  local fedcsap_bottom_q="$3"
  local dirichlet_alpha="$4"
  local attack_method="$5"
  local mal_pcnt="$6"
  local aggregation_method="$7"

  local cmd=(
    python main.py
    --type="${type}"
    --aggregation_methods="${aggregation_method}"
    --attack_methods="${attack_method}"
    --mal_pcnt="${mal_pcnt}"
    --resumed_model=false
    --epochs=200
    --number_of_total_participants=25
    --committee_size=5
    --no_models=20
    --noniid=sampling_dirichlet
    --dirichlet_alpha="${dirichlet_alpha}"
    --eta=0.1
    --fedcsap_bottom_q="${fedcsap_bottom_q}"
    --seed=0
    --"${run_tag}"
  )

  if [[ "${DRY_RUN}" == "1" ]]; then
    printf 'DRY_RUN: %q ' "${cmd[@]}"
    echo
    return 0
  fi

  nohup "${cmd[@]}" > /dev/null 2>&1 &
  echo "Started ${run_tag}: q=${fedcsap_bottom_q}, attack=${attack_method}, mal_pcnt=${mal_pcnt}, aggregation=${aggregation_method}, pid=$!"
}

group_index=1
jobs_in_group=0
selected_count=0

while IFS=, read -r run_tag type fedcsap_bottom_q dirichlet_alpha attack_method mal_pcnt aggregation_method note; do
  # skip header
  if [[ "${run_tag}" == "run_tag" ]]; then
    continue
  fi

  run_num=$(normalize_run_id "${run_tag}")
  if (( run_num < START_ID || run_num > END_ID )); then
    continue
  fi

  if (( jobs_in_group == 0 )); then
    echo "===== Group ${group_index} started (max parallel: ${MAX_PARALLEL}) ====="
  fi

  start_run "${run_tag}" "${type}" "${fedcsap_bottom_q}" "${dirichlet_alpha}" "${attack_method}" "${mal_pcnt}" "${aggregation_method}"
  selected_count=$((selected_count + 1))
  jobs_in_group=$((jobs_in_group + 1))

  if (( jobs_in_group == MAX_PARALLEL )); then
    if [[ "${DRY_RUN}" != "1" ]]; then
      echo "===== Group ${group_index} waiting for ${jobs_in_group} job(s) to finish ====="
      wait
      echo "===== Group ${group_index} finished ====="
    else
      echo "===== Group ${group_index} dry-run finished (${jobs_in_group} job(s)) ====="
    fi
    group_index=$((group_index + 1))
    jobs_in_group=0
  fi
done < "${CONFIG_FILE}"

if (( jobs_in_group > 0 )); then
  if [[ "${DRY_RUN}" != "1" ]]; then
    echo "===== Group ${group_index} waiting for remaining ${jobs_in_group} job(s) to finish ====="
    wait
    echo "===== Group ${group_index} finished ====="
  else
    echo "===== Group ${group_index} dry-run finished (${jobs_in_group} job(s)) ====="
  fi
fi

if (( selected_count == 0 )); then
  echo "No runs selected in [${START_INPUT}, ${END_INPUT}] from ${CONFIG_FILE}" >&2
  exit 1
fi

echo "Done. Selected runs: ${selected_count}, range=[${START_INPUT}, ${END_INPUT}]"
