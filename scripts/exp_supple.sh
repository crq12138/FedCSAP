#!/usr/bin/env bash
set -euo pipefail

# 补实验：重跑失败的 run（从 CSV 读取）。
#
# 用法：
#   bash scripts/exp_supple.sh
#
# 可选环境变量：
#   CONFIG_FILE=scripts/configs/exp_supple_failed_runs.csv
#   MAX_PARALLEL=3
#   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#   DRY_RUN=1

CONFIG_FILE=${CONFIG_FILE:-scripts/configs/exp_supple_failed_runs.csv}
MAX_PARALLEL=${MAX_PARALLEL:-3}
PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
DRY_RUN=${DRY_RUN:-0}
export PYTORCH_CUDA_ALLOC_CONF

if ! [[ "${MAX_PARALLEL}" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_PARALLEL must be a positive integer, got: ${MAX_PARALLEL}" >&2
  exit 1
fi

if [[ ! -f "${CONFIG_FILE}" ]]; then
  echo "Config file not found: ${CONFIG_FILE}" >&2
  exit 1
fi

adversary_count_for_level() {
  case "$1" in
    0.3) echo 8 ;;
    0.2) echo 5 ;;
    0.1) echo 2 ;;
    *)
      echo "Unsupported mal_pcnt: $1" >&2
      exit 1
      ;;
  esac
}

epochs_for_run() {
  local type="$1"
  local attack_method="$2"

  if [[ "${attack_method}" == "dba" ]]; then
    if [[ "${type}" == "mnist" ]]; then
      echo 100
    else
      echo 200
    fi
    return 0
  fi

  if [[ "${type}" == "pathmnist" ]]; then
    echo 150
  else
    echo 200
  fi
}


prepare_run_folder() {
  local run_tag="$1"
  local run_folder="runs/${run_tag}"

  rm -rf "${run_folder}"
  mkdir -p "${run_folder}"
}

start_run() {
  local run_tag="$1"
  local type="$2"
  local attack_method="$3"
  local mal_pcnt="$4"
  local aggregation_method="$5"
  local dirichlet_alpha="$6"

  local adversary_count
  local epochs
  adversary_count=$(adversary_count_for_level "${mal_pcnt}")
  epochs=$(epochs_for_run "${type}" "${attack_method}")

  local cmd=(
    python main.py
    --type="${type}"
    --aggregation_methods="${aggregation_method}"
    --attack_methods="${attack_method}"
    --"number_of_adversary_${attack_method}"="${adversary_count}"
    --mal_pcnt="${mal_pcnt}"
    --resumed_model=false
    --epochs="${epochs}"
    --number_of_total_participants=25
    --committee_size=5
    --no_models=20
    --noniid=sampling_dirichlet
    --dirichlet_alpha="${dirichlet_alpha}"
    --eta=0.1
    --fedcsap_bottom_q=0.2
    --committee_election=reputation
    --seed=0
    --"${run_tag}"
  )

  if [[ "${aggregation_method}" == "flshield" ]]; then
    cmd+=(--bijective_flshield)
  fi

  if [[ "${attack_method}" == "dba" ]]; then
    local poisoning_per_batch=10
    if [[ "${aggregation_method}" == "fedcsap" ]]; then
      poisoning_per_batch=5
    elif [[ "${aggregation_method}" == "median" || "${aggregation_method}" == "krum" ]]; then
      poisoning_per_batch=15
    fi
    cmd+=(--poisoning_per_batch="${poisoning_per_batch}")
  fi
  
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf 'DRY_RUN: %q ' "${cmd[@]}"
    echo
    return 0
  fi


  prepare_run_folder "${run_tag}"

  nohup "${cmd[@]}" > /dev/null 2>&1 &
  echo "Started ${run_tag}: type=${type}, attack=${attack_method}, mal_pcnt=${mal_pcnt}, aggregation=${aggregation_method}, epochs=${epochs}, pid=$!"
}

group_index=1
jobs_in_group=0
selected_count=0

while IFS=, read -r run_tag type attack_method mal_pcnt aggregation_method dirichlet_alpha; do
  if [[ "${run_tag}" == "run_tag" ]]; then
    continue
  fi

  dirichlet_alpha=${dirichlet_alpha:-0.9}

  if (( jobs_in_group == 0 )); then
    echo "===== Group ${group_index} started (max parallel: ${MAX_PARALLEL}) ====="
  fi

  start_run "${run_tag}" "${type}" "${attack_method}" "${mal_pcnt}" "${aggregation_method}" "${dirichlet_alpha}"
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
  echo "No runs selected from ${CONFIG_FILE}" >&2
  exit 1
fi

echo "Done. Selected runs: ${selected_count}"
