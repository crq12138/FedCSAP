#!/usr/bin/env bash
set -euo pipefail

# 实验三：委员会占领攻击实验（run02/run08）
# 用法：
#   bash scripts/exp_03.sh

start_run() {
  local run_tag="$1"
  local committee_election="$2"

  nohup python main.py \
    --type=cifar \
    --aggregation_methods=fedcsap \
    --attack_methods=targeted_label_flip \
    --mal_pcnt=0.3 \
    --resumed_model=false \
    --epochs=200 \
    --number_of_total_participants=25 \
    --committee_size=5 \
    --no_models=20 \
    --noniid=sampling_dirichlet \
    --fedcsap_bottom_q=0.2 \
    --committee_election="${committee_election}" \
    --eta=0.1 \
    --seed=0 \
    --"${run_tag}" \
    > /dev/null 2>&1 &

  echo "Started ${run_tag}: committee_election=${committee_election}, pid=$!"
}

# reputation 方案（对照）
# start_run run_02 reputation
# random 方案（目标）
start_run run_008 random

echo "All jobs started in background."