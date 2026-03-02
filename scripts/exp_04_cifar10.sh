#!/usr/bin/env bash
set -euo pipefail

# 实验四：常规恶意攻击防御能力（CIFAR10）
# 攻击方式：tlfa / IPMA / SF
# 攻击比例：0.3 / 0.2 / 0.1
# 聚合方式：fedcsap / flshield / fltrust / krum / AFA / FedAvg / median / foolsgold
# 用法：
#   bash data/exp_04_cifar10
# 可选：
#   MAX_PARALLEL=3 bash scripts/exp_04_cifar10.sh
#   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True bash scripts/exp_04_cifar10.sh

MAX_PARALLEL=${MAX_PARALLEL:-3}
PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export PYTORCH_CUDA_ALLOC_CONF

if ! [[ "${MAX_PARALLEL}" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_PARALLEL must be a positive integer, got: ${MAX_PARALLEL}" >&2
  exit 1
fi

start_run() {
  local run_tag="$1"
  local attack_method="$2"
  local mal_pcnt="$3"
  local aggregation_method="$4"
  local adversary_count="$5"

  nohup python main.py \
    --type=cifar \
    --aggregation_methods="${aggregation_method}" \
    --attack_methods="${attack_method}" \
    --"number_of_adversary_${attack_method}"="${adversary_count}" \
    --mal_pcnt="${mal_pcnt}" \
    --resumed_model=false \
    --epochs=200 \
    --number_of_total_participants=25 \
    --committee_size=5 \
    --no_models=20 \
    --noniid=sampling_dirichlet \
    --eta=0.1 \
    --fedcsap_bottom_q=0.2 \
    --committee_election=reputation \
    --seed=0 \
    --"${run_tag}" \
    > /dev/null 2>&1 &

  echo "Started ${run_tag}: attack=${attack_method}, mal_pcnt=${mal_pcnt}, aggregation=${aggregation_method}, adversary_count=${adversary_count}, pid=$!"
}

aggregations=(fedcsap flshield fltrust krum afa fedavg median foolsgold)
levels=(0.3 0.2 0.1)
attacks=(targeted_label_flip inner_product_manipulation sf)

# 按实验模板：25 * 比例，四舍五入取整
# 0.3 -> 8, 0.2 -> 5, 0.1 -> 3
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

run_id=9
group_index=1
jobs_in_group=0

for attack in "${attacks[@]}"; do
  for aggr in "${aggregations[@]}"; do
    for level in "${levels[@]}"; do
      run_tag=$(printf 'run_%03d' "${run_id}")
      adversary_count=$(adversary_count_for_level "${level}")

      if (( jobs_in_group == 0 )); then
        echo "===== Group ${group_index} started (max parallel: ${MAX_PARALLEL}) ====="
      fi

      start_run "${run_tag}" "${attack}" "${level}" "${aggr}" "${adversary_count}"
      run_id=$((run_id + 1))
      jobs_in_group=$((jobs_in_group + 1))

      if (( jobs_in_group == MAX_PARALLEL )); then
        echo "===== Group ${group_index} waiting for ${jobs_in_group} job(s) to finish ====="
        wait
        echo "===== Group ${group_index} finished ====="
        group_index=$((group_index + 1))
        jobs_in_group=0
      fi
    done
  done
done

if (( jobs_in_group > 0 )); then
  echo "===== Group ${group_index} waiting for remaining ${jobs_in_group} job(s) to finish ====="
  wait
  echo "===== Group ${group_index} finished ====="
fi

echo "All jobs completed. Total runs: $((run_id - 9))"
