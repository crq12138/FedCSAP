#!/usr/bin/env bash
set -euo pipefail

# Run FedCSAP experiments in parallel to sweep fedcsap_bottom_q.
# Usage:
#   bash scripts/run_fedcsap_bottom_q_sweep.sh

qs=(0.1 0.2 0.3 0.4)
runs=(run_001 run_002 run_003 run_004)

for i in "${!qs[@]}"; do
  q="${qs[$i]}"
  run_tag="${runs[$i]}"

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
    --eta=0.1 \
    --fedcsap_bottom_q="${q}" \
    --seed=0 \
    --"${run_tag}" \
    > /dev/null 2>&1 &

  echo "Started q=${q}, run=${run_tag}, pid=$!"
done

echo "All jobs started in background."
