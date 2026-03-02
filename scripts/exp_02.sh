#!/usr/bin/env bash
set -euo pipefail

# 实验二：信誉管理实验（run05/run06/run07）
# 用法：
#   bash scripts/exp_02.sh

start_run() {
  local run_tag="$1"
  local data_type="$2"
  local attack_method="$3"
  local adversary_count="$4"

  nohup python main.py \
    --type="${data_type}" \
    --aggregation_methods=fedcsap \
    --attack_methods="${attack_method}" \
    --"number_of_adversary_${attack_method}"="${adversary_count}" \
    --mal_pcnt=0.3 \
    --resumed_model=false \
    --epochs=200 \
    --number_of_total_participants=25 \
    --committee_size=5 \
    --no_models=20 \
    --noniid=sampling_dirichlet \
    --eta=0.1 \
    --fedcsap_bottom_q=0.2 \
    --seed=0 \
    --"${run_tag}" \
    > /dev/null 2>&1 &

  echo "Started ${run_tag}: type=${data_type}, attack=${attack_method}, adversary_count=${adversary_count}, pid=$!"
}

# 25 * 0.3 = 7.5，按模板中的 round 逻辑取 8
start_run run_005 pathmnist targeted_label_flip 8
# start_run run_006 cifar inner_product_manipulation 8
start_run run_007 pathmnist inner_product_manipulation 8

echo "All jobs started in background."
