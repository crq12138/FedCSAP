#!/usr/bin/env bash
set -euo pipefail

# 实验五：复杂攻击场景（8个恶意参与方，TLF/SF/IPM/DBA 各2个）
#
# 21次执行 = 7种聚合方案 × 3种数据集（CIFAR10/PATHMNIST/MNIST）
# 默认读取 scripts/config/exp_05_complex_attack/runs.csv
#
# 用法：
#   bash scripts/exp_05_complex_attack.sh
#   bash scripts/exp_05_complex_attack.sh run_380 run_386
#
# 可选环境变量：
#   CONFIG_FILE=scripts/config/exp_05_complex_attack/runs.csv
#   MAX_PARALLEL=3
#   DRY_RUN=1
#   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#   DBA_POISONING_PER_BATCH=60

CONFIG_FILE=${CONFIG_FILE:-scripts/configs/exp_05_complex_attack_runs.csv}
MAX_PARALLEL=${MAX_PARALLEL:-3}
DRY_RUN=${DRY_RUN:-0}
PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
DBA_POISONING_PER_BATCH=${DBA_POISONING_PER_BATCH:-60}
export PYTORCH_CUDA_ALLOC_CONF

if [[ ! -f "${CONFIG_FILE}" ]]; then
  echo "Config file not found: ${CONFIG_FILE}" >&2
  exit 1
fi

if ! [[ "${MAX_PARALLEL}" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_PARALLEL must be a positive integer, got: ${MAX_PARALLEL}" >&2
  exit 1
fi

normalize_run_id() {
  local input="$1"
  if [[ "${input}" =~ ^run_([0-9]+)$ ]]; then
    echo $((10#${BASH_REMATCH[1]}))
  elif [[ "${input}" =~ ^([0-9]+)$ ]]; then
    echo $((10#${BASH_REMATCH[1]}))
  else
    echo "Invalid run id: ${input}. Use 373 or run_373 format." >&2
    exit 1
  fi
}

START_INPUT=${1:-run_373}
END_INPUT=${2:-run_393}
START_ID=$(normalize_run_id "${START_INPUT}")
END_ID=$(normalize_run_id "${END_INPUT}")

if (( START_ID > END_ID )); then
  echo "Start run must be <= end run. Got: ${START_INPUT} .. ${END_INPUT}" >&2
  exit 1
fi

epochs_for_dataset() {
  case "$1" in
    cifar) echo 210 ;;
    pathmnist) echo 150 ;;
    mnist) echo 200 ;;
    *)
      echo "Unsupported dataset type: $1" >&2
      exit 1
      ;;
  esac
}

start_run() {
  local run_tag="$1"
  local type="$2"
  local aggregation_method="$3"
  local epochs
  epochs=$(epochs_for_dataset "${type}")

  local cmd=(
    python main.py
    --type="${type}"
    --aggregation_methods="${aggregation_method}"
    --attack_methods=mixed_8_tlf_sf_ipm_dba
    --number_of_adversary_mixed_8_tlf_sf_ipm_dba=8
    --number_of_adversary_targeted_label_flip=2
    --number_of_adversary_sf=2
    --number_of_adversary_inner_product_manipulation=2
    --number_of_adversary_dba=2
    --tlf_label=medium
    --mal_pcnt=0.32
    --poisoning_per_batch="${DBA_POISONING_PER_BATCH}"
    --resumed_model=false
    --epochs="${epochs}"
    --number_of_total_participants=25
    --committee_size=5
    --no_models=20
    --noniid=sampling_dirichlet
    --dirichlet_alpha=0.9
    --eta=0.1
    --fedcsap_bottom_q=0.2
    --committee_election=reputation
    --seed=0
    --complex_attack_mode=mixed_8_tlf_sf_ipm_dba
    --"${run_tag}"
  )

  if [[ "${aggregation_method}" == "flshield" ]]; then
    cmd+=(--bijective_flshield)
  fi

  if [[ "${DRY_RUN}" == "1" ]]; then
    printf 'DRY_RUN: %q ' "${cmd[@]}"
    echo
    return 0
  fi

  nohup "${cmd[@]}" > /dev/null 2>&1 &
  echo "Started ${run_tag}: type=${type}, aggregation=${aggregation_method}, mixed_adv={TLF:2,SF:2,IPM:2,DBA:2}, pid=$!"
}

group_index=1
jobs_in_group=0
selected_count=0

while IFS=, read -r run_tag type aggregation_method; do
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

  start_run "${run_tag}" "${type}" "${aggregation_method}"
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
