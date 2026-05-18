#!/usr/bin/env bash
set -euo pipefail

# 实验九：复杂攻击场景下 FedCSAP 三数据集模块耗时统计
# 参数与 exp_05_complex_attack.sh 中 FedCSAP 方案一致

CONFIG_FILE=${CONFIG_FILE:-scripts/configs/exp_09_time_runs.csv}
MAX_PARALLEL=${MAX_PARALLEL:-3}
DRY_RUN=${DRY_RUN:-0}
PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
DBA_POISONING_PER_BATCH=${DBA_POISONING_PER_BATCH:-60}
COMMITTEE_ELECTION=${COMMITTEE_ELECTION:-reputation}
export PYTORCH_CUDA_ALLOC_CONF

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
    echo "Invalid run id: ${input}" >&2
    exit 1
  fi
}

epochs_for_dataset() {
  case "$1" in
    cifar) echo 200 ;;
    pathmnist) echo 150 ;;
    mnist) echo 100 ;;
    *) echo "Unsupported dataset type: $1" >&2; exit 1 ;;
  esac
}

START_INPUT=${1:-run_484}
END_INPUT=${2:-run_486}
START_ID=$(normalize_run_id "${START_INPUT}")
END_ID=$(normalize_run_id "${END_INPUT}")

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
    --seed=0
    --complex_attack_mode=mixed_8_tlf_sf_ipm_dba
    --committee_election="${COMMITTEE_ELECTION}"
    --"${run_tag}"
  )

  if [[ "${DRY_RUN}" == "1" ]]; then
    printf 'DRY_RUN: %q ' "${cmd[@]}"; echo
    return 0
  fi

  nohup "${cmd[@]}" > /dev/null 2>&1 &
  echo "Started ${run_tag}: type=${type}, aggregation=${aggregation_method}, pid=$!"
}

group_index=1
jobs_in_group=0
selected_count=0

while IFS=, read -r run_tag type aggregation_method; do
  [[ "${run_tag}" == "run_tag" ]] && continue
  run_num=$(normalize_run_id "${run_tag}")
  (( run_num < START_ID || run_num > END_ID )) && continue

  (( jobs_in_group == 0 )) && echo "===== Group ${group_index} started (max parallel: ${MAX_PARALLEL}) ====="
  start_run "${run_tag}" "${type}" "${aggregation_method}"
  selected_count=$((selected_count + 1))
  jobs_in_group=$((jobs_in_group + 1))

  if (( jobs_in_group == MAX_PARALLEL )); then
    if [[ "${DRY_RUN}" != "1" ]]; then wait; fi
    echo "===== Group ${group_index} finished ====="
    group_index=$((group_index + 1))
    jobs_in_group=0
  fi
done < "${CONFIG_FILE}"

if (( jobs_in_group > 0 )); then
  if [[ "${DRY_RUN}" != "1" ]]; then wait; fi
  echo "===== Group ${group_index} finished ====="
fi

echo "Done. Selected runs: ${selected_count}, range=[${START_INPUT}, ${END_INPUT}]"
