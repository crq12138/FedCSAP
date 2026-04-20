#!/usr/bin/env bash
set -euo pipefail

# 实验八：CBRFL 对比实验（MNIST/PATHMNIST/CIFAR10 × TLF/SF/IPM/DBA × 强度2/5/8）
#
# 36次执行 = 3种数据集 × 4种攻击 × 3种攻击强度
# 默认读取 scripts/configs/exp_08_cbrfl_runs.csv
#
# 用法：
#   bash scripts/exp_08_cbrfl.sh
#   bash scripts/exp_08_cbrfl.sh run_445 run_480
#   bash scripts/exp_08_cbrfl.sh run_476 run_477 --ipm_val=-1.0
#
# 可选环境变量：
#   CONFIG_FILE=scripts/configs/exp_08_cbrfl_runs.csv
#   MAX_PARALLEL=3
#   DRY_RUN=1
#   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#   POISONING_PER_BATCH=60
#   COMMITTEE_ELECTION=random
#   NUMBER_OF_TOTAL_PARTICIPANTS=25
#   COMMITTEE_SIZE=10
#   NO_MODELS=15

CONFIG_FILE=${CONFIG_FILE:-scripts/configs/exp_08_cbrfl_runs.csv}
MAX_PARALLEL=${MAX_PARALLEL:-3}
DRY_RUN=${DRY_RUN:-0}
PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
POISONING_PER_BATCH=${POISONING_PER_BATCH:-60}
COMMITTEE_ELECTION=${COMMITTEE_ELECTION:-random}
NUMBER_OF_TOTAL_PARTICIPANTS=${NUMBER_OF_TOTAL_PARTICIPANTS:-25}
COMMITTEE_SIZE=${COMMITTEE_SIZE:-10}
NO_MODELS=${NO_MODELS:-15}
TRAIN_GRAD_CLIP=${TRAIN_GRAD_CLIP:-10}
POISON_GRAD_CLIP=${POISON_GRAD_CLIP:-5}
SF_SCALE=${SF_SCALE:-0.5}
export PYTORCH_CUDA_ALLOC_CONF

if [[ ! -f "${CONFIG_FILE}" ]]; then
  echo "Config file not found: ${CONFIG_FILE}" >&2
  exit 1
fi

if ! [[ "${MAX_PARALLEL}" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_PARALLEL must be a positive integer, got: ${MAX_PARALLEL}" >&2
  exit 1
fi

if ! [[ "${POISONING_PER_BATCH}" =~ ^[1-9][0-9]*$ ]]; then
  echo "POISONING_PER_BATCH must be a positive integer, got: ${POISONING_PER_BATCH}" >&2
  exit 1
fi

if ! [[ "${NUMBER_OF_TOTAL_PARTICIPANTS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "NUMBER_OF_TOTAL_PARTICIPANTS must be a positive integer, got: ${NUMBER_OF_TOTAL_PARTICIPANTS}" >&2
  exit 1
fi

if ! [[ "${COMMITTEE_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "COMMITTEE_SIZE must be a positive integer, got: ${COMMITTEE_SIZE}" >&2
  exit 1
fi

if ! [[ "${NO_MODELS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "NO_MODELS must be a positive integer, got: ${NO_MODELS}" >&2
  exit 1
fi

if (( COMMITTEE_SIZE + NO_MODELS != NUMBER_OF_TOTAL_PARTICIPANTS )); then
  echo "COMMITTEE_SIZE + NO_MODELS must equal NUMBER_OF_TOTAL_PARTICIPANTS. Got ${COMMITTEE_SIZE} + ${NO_MODELS} != ${NUMBER_OF_TOTAL_PARTICIPANTS}" >&2
  exit 1
fi

normalize_run_id() {
  local input="$1"
  if [[ "${input}" =~ ^run_([0-9]+)$ ]]; then
    echo $((10#${BASH_REMATCH[1]}))
  elif [[ "${input}" =~ ^([0-9]+)$ ]]; then
    echo $((10#${BASH_REMATCH[1]}))
  else
    echo "Invalid run id: ${input}. Use 445 or run_445 format." >&2
    exit 1
  fi
}

START_INPUT=${1:-run_445}
END_INPUT=${2:-run_480}
START_ID=$(normalize_run_id "${START_INPUT}")
END_ID=$(normalize_run_id "${END_INPUT}")
EXTRA_ARGS=("${@:3}")

if (( START_ID > END_ID )); then
  echo "Start run must be <= end run. Got: ${START_INPUT} .. ${END_INPUT}" >&2
  exit 1
fi

epochs_for_dataset() {
  case "$1" in
    mnist) echo 100 ;;
    pathmnist) echo 150 ;;
    cifar) echo 200 ;;
    *)
      echo "Unsupported dataset type: $1" >&2
      exit 1
      ;;
  esac
}

attack_method_from_code() {
  case "$1" in
    TLF) echo targeted_label_flip ;;
    SF) echo sf ;;
    IPM) echo inner_product_manipulation ;;
    DBA) echo dba ;;
    *)
      echo "Unsupported attack code: $1. Allowed: TLF/SF/IPM/DBA" >&2
      exit 1
      ;;
  esac
}

mal_pcnt_for_strength() {
  case "$1" in
    2) echo 0.1 ;;
    5) echo 0.2 ;;
    8) echo 0.3 ;;
    *)
      echo "Unsupported attack_strength: $1. Allowed: 2/5/8" >&2
      exit 1
      ;;
  esac
}

start_run() {
  local run_tag="$1"
  local type="$2"
  local attack_code="$3"
  local attack_strength="$4"
  local aggregation_method="$5"

  local attack_method
  attack_method=$(attack_method_from_code "${attack_code}")

  local mal_pcnt
  mal_pcnt=$(mal_pcnt_for_strength "${attack_strength}")

  local epochs
  epochs=$(epochs_for_dataset "${type}")

  local cmd=(
    python main.py
    --type="${type}"
    --aggregation_methods="${aggregation_method}"
    --attack_methods="${attack_method}"
    --"number_of_adversary_${attack_method}"="${attack_strength}"
    --mal_pcnt="${mal_pcnt}"
    --poisoning_per_batch="${POISONING_PER_BATCH}"
    --resumed_model=false
    --epochs="${epochs}"
    --number_of_total_participants="${NUMBER_OF_TOTAL_PARTICIPANTS}"
    --committee_size="${COMMITTEE_SIZE}"
    --no_models="${NO_MODELS}"
    --noniid=sampling_dirichlet
    --dirichlet_alpha=0.9
    --eta=0.1
    --committee_election="${COMMITTEE_ELECTION}"
    --train_grad_clip="${TRAIN_GRAD_CLIP}"
    --poison_grad_clip="${POISON_GRAD_CLIP}"
    --seed=0
    --"${run_tag}"
  )

  if (( ${#EXTRA_ARGS[@]} > 0 )); then
    cmd+=("${EXTRA_ARGS[@]}")
  fi

  if [[ "${attack_method}" == "sf" ]]; then
    cmd+=(--sf_scale="${SF_SCALE}")
  fi

  if [[ "${aggregation_method}" == "flshield" ]]; then
    cmd+=(--bijective_flshield)
  fi

  if [[ "${DRY_RUN}" == "1" ]]; then
    printf 'DRY_RUN: %q ' "${cmd[@]}"
    echo
    return 0
  fi

  nohup "${cmd[@]}" > /dev/null 2>&1 &
  echo "Started ${run_tag}: type=${type}, attack=${attack_code}, attack_method=${attack_method}, strength=${attack_strength}, mal_pcnt=${mal_pcnt}, aggregation=${aggregation_method}, pid=$!"
}

group_index=1
jobs_in_group=0
selected_count=0

while IFS=, read -r run_tag type attack_code attack_strength aggregation_method; do
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

  start_run "${run_tag}" "${type}" "${attack_code}" "${attack_strength}" "${aggregation_method}"
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
