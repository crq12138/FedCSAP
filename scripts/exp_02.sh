#!/usr/bin/env bash
set -euo pipefail

# 实验二：信誉管理实验（run05/run06/run07）
# 用法：
#   bash scripts/exp_02.sh

start_run() {
  local run_tag="$1"
  local data_type="$2"
  local attack_method="$3"

  nohup python main.py \
    --type="${data_type}" \
    --aggregation_methods=fedcsap \
    --attack_methods="${attack_method}" \
    --mal_pcnt=0.3 \
    --resumed_model=false \
    --epochs=150 \
    --number_of_total_participants=25 \
    --committee_size=5 \
    --no_models=20 \
    --noniid=sampling_dirichlet \
    --fedcsap_bottom_q=0.2 \
    --seed=0 \
    --"${run_tag}" \
    > /dev/null 2>&1 &

  echo "Started ${run_tag}: type=${data_type}, attack=${attack_method}, pid=$!"
}

start_run run_05 pathmnist targeted_label_flip
start_run run_06 cifar inner_product_manipulation
start_run run_07 pathmnist inner_product_manipulation

echo "All jobs started in background."
