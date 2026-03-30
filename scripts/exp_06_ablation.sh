#!/usr/bin/env bash
set -euo pipefail

# 实验六：复杂攻击场景消融实验（8个恶意参与方，TLF/SF/IPM/DBA 各2个）
#
# 12次执行 = 4种配置 × 3种数据集（CIFAR10/PATHMNIST/MNIST）
# 配置矩阵：
#   1) 委员会治理 ×，细粒度鲁棒聚合 ×  -> aggregation=fedavg, committee_election=random
#   2) 委员会治理 √，细粒度鲁棒聚合 ×  -> aggregation=fedavg, committee_election=reputation
#   3) 委员会治理 ×，细粒度鲁棒聚合 √  -> aggregation=fedcsap, committee_election=random
#   4) 委员会治理 √，细粒度鲁棒聚合 √  -> aggregation=fedcsap, committee_election=reputation
#
# 复杂攻击默认附加“委员会占领后梯度翻转”能力：
#   --fedcsap_committee_takeover_attack=true
#
# 默认读取 scripts/configs/exp_06_ablation_runs.csv
#
# 用法：
#   bash scripts/exp_06_ablation.sh
#   bash scripts/exp_06_ablation.sh run_394 run_405
#
# 可选环境变量：
#   CONFIG_FILE=scripts/configs/exp_06_ablation_runs.csv
#   MAX_PARALLEL=3
#   DRY_RUN=1
#   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#   DBA_POISONING_PER_BATCH=60

CONFIG_FILE=${CONFIG_FILE:-scripts/configs/exp_06_ablation_runs.csv}
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
    echo "Invalid run id: ${input}. Use 394 or run_394 format." >&2
    exit 1
  fi
}

START_INPUT=${1:-run_394}
END_INPUT=${2:-run_405}
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
  local committee_election="$4"
  local takeover_attack_switch="$5"
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
    --committee_election="${committee_election}"
    --fedcsap_committee_takeover_attack="${takeover_attack_switch}"
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
  echo "Started ${run_tag}: type=${type}, aggregation=${aggregation_method}, committee_election=${committee_election}, committee_takeover_attack=${takeover_attack_switch}, mixed_adv={TLF:2,SF:2,IPM:2,DBA:2}, pid=$!"
}

group_index=1
jobs_in_group=0
selected_count=0

while IFS=, read -r run_tag type aggregation_method committee_election fedcsap_committee_takeover_attack; do
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

  start_run "${run_tag}" "${type}" "${aggregation_method}" "${committee_election}" "${fedcsap_committee_takeover_attack}"
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
