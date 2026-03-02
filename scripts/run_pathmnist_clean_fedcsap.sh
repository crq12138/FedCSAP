#!/usr/bin/env bash
set -euo pipefail

# 运行一次「干净 PathMNIST + FedCSAP」实验，用于排查训练/精度问题。
# 特点：
#   1) 不启用投毒训练（--is_poison=false）
#   2) 无恶意客户端（--mal_pcnt=0 且 adversary 数量为 0）
#   3) 聚合方法固定为 fedcsap
#
# 用法：
#   bash scripts/run_pathmnist_clean_fedcsap.sh
#   EPOCHS=50 TOTAL_PARTICIPANTS=25 COMMITTEE_SIZE=5 NO_MODELS=20 \
#     RUN_TAG=run_clean_pathmnist_fedcsap bash scripts/run_pathmnist_clean_fedcsap.sh

EPOCHS="${EPOCHS:-50}"
TOTAL_PARTICIPANTS="${TOTAL_PARTICIPANTS:-25}"
COMMITTEE_SIZE="${COMMITTEE_SIZE:-5}"
NO_MODELS="${NO_MODELS:-20}"
FEDCSAP_BOTTOM_Q="${FEDCSAP_BOTTOM_Q:-0.2}"
LR_ETA="${LR_ETA:-0.1}"
SEED="${SEED:-0}"
RUN_TAG="${RUN_TAG:-run_clean_pathmnist_fedcsap}"

if [[ "${RUN_TAG}" != run_* ]]; then
  echo "[ERROR] RUN_TAG 必须以 run_ 开头，例如 run_clean_pathmnist_fedcsap"
  exit 1
fi

mkdir -p "runs/${RUN_TAG}"

echo "[INFO] Starting clean PathMNIST FedCSAP run"
echo "[INFO] RUN_TAG=${RUN_TAG}, epochs=${EPOCHS}, participants=${TOTAL_PARTICIPANTS}, committee=${COMMITTEE_SIZE}, no_models=${NO_MODELS}"

after_cmd=(
  python main.py
  --type=pathmnist
  --aggregation_methods=fedcsap
  --attack_methods=targeted_label_flip
  --is_poison=false
  --mal_pcnt=0
  --number_of_adversary_targeted_label_flip=0
  --resumed_model=false
  --epochs="${EPOCHS}"
  --number_of_total_participants="${TOTAL_PARTICIPANTS}"
  --committee_size="${COMMITTEE_SIZE}"
  --no_models="${NO_MODELS}"
  --noniid=iid
  --eta="${LR_ETA}"
  --fedcsap_bottom_q="${FEDCSAP_BOTTOM_Q}"
  --seed="${SEED}"
  --"${RUN_TAG}"
)

printf '[CMD]'; printf ' %q' "${after_cmd[@]}"; printf '\n'
"${after_cmd[@]}"

echo "[INFO] Run finished. Check outputs under runs/${RUN_TAG}/"
