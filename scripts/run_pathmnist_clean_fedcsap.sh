#!/usr/bin/env bash
set -euo pipefail

# 运行一次「干净 PathMNIST + FedCSAP」实验，并在结束后输出简要诊断。
# 目标：排查“训练几乎不动 / 精度不提升”的问题。
#
# 关键策略：
#   1) 禁用投毒：--is_poison=false, --mal_pcnt=0, --number_of_adversary_targeted_label_flip=0
#   2) 使用 FedCSAP：--aggregation_methods=fedcsap
#   3) 诊断友好默认值：
#      - eta 默认 1.0（PathMNIST 默认更接近该值，避免全局更新过小）
#      - lr 默认 0.01（比 0.1 更稳）
#      - committee_validation_sample_ratio 默认 1.0（减少验证抽样噪声）
#      - committee_election 默认 random（绕开冷启动信誉对选举的影响）
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
LR_ETA="${LR_ETA:-1.0}"
LR="${LR:-0.01}"
SEED="${SEED:-0}"
VAL_RATIO="${VAL_RATIO:-1.0}"
COMMITTEE_ELECTION="${COMMITTEE_ELECTION:-random}"
RUN_TAG="${RUN_TAG:-run_clean_pathmnist_fedcsap}"

if [[ "${RUN_TAG}" != run_* ]]; then
  echo "[ERROR] RUN_TAG 必须以 run_ 开头，例如 run_clean_pathmnist_fedcsap"
  exit 1
fi

mkdir -p "runs/${RUN_TAG}"

echo "[INFO] Starting clean PathMNIST FedCSAP run"
echo "[INFO] RUN_TAG=${RUN_TAG}, epochs=${EPOCHS}, participants=${TOTAL_PARTICIPANTS}, committee=${COMMITTEE_SIZE}, no_models=${NO_MODELS}, lr=${LR}, eta=${LR_ETA}"

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
  --lr="${LR}"
  --number_of_total_participants="${TOTAL_PARTICIPANTS}"
  --committee_size="${COMMITTEE_SIZE}"
  --no_models="${NO_MODELS}"
  --noniid=iid
  --eta="${LR_ETA}"
  --fedcsap_bottom_q="${FEDCSAP_BOTTOM_Q}"
  --committee_validation_sample_ratio="${VAL_RATIO}"
  --committee_election="${COMMITTEE_ELECTION}"
  --minimize_logging=false
  --seed="${SEED}"
  --"${RUN_TAG}"
)

printf '[CMD]'; printf ' %q' "${after_cmd[@]}"; printf '\n'
"${after_cmd[@]}"

echo "[INFO] Run finished. Check outputs under runs/${RUN_TAG}/"

GLOBAL_CSV="runs/${RUN_TAG}/global_metrics.csv"
CLIENT_CSV="runs/${RUN_TAG}/fedcsap_client_metrics.csv"

if [[ -f "${GLOBAL_CSV}" ]]; then
  python - <<'PY'
import csv
import os

run_tag = os.environ.get('RUN_TAG', 'run_clean_pathmnist_fedcsap')
global_csv = f'runs/{run_tag}/global_metrics.csv'

rows = []
with open(global_csv, 'r', newline='') as f:
    reader = csv.DictReader(f)
    for r in reader:
        rows.append(r)

if not rows:
    print('[DIAG] global_metrics.csv is empty.')
    raise SystemExit(0)

start_acc = float(rows[0]['global_acc'])
end_acc = float(rows[-1]['global_acc'])
start_f1 = float(rows[0]['global_macro_f1'])
end_f1 = float(rows[-1]['global_macro_f1'])

delta_acc = end_acc - start_acc
delta_f1 = end_f1 - start_f1

print(f'[DIAG] global_acc: start={start_acc:.4f}, end={end_acc:.4f}, delta={delta_acc:.4f}')
print(f'[DIAG] global_macro_f1: start={start_f1:.4f}, end={end_f1:.4f}, delta={delta_f1:.4f}')
if abs(delta_acc) < 0.5:
    print('[DIAG][WARN] Accuracy change < 0.5, learning appears stalled.')
PY
else
  echo "[DIAG][WARN] ${GLOBAL_CSV} not found; skip global metric diagnosis."
fi

if [[ -f "${CLIENT_CSV}" ]]; then
  python - <<'PY'
import csv
import os
from collections import defaultdict

run_tag = os.environ.get('RUN_TAG', 'run_clean_pathmnist_fedcsap')
client_csv = f'runs/{run_tag}/fedcsap_client_metrics.csv'

selected_per_epoch = defaultdict(int)
total_per_epoch = defaultdict(int)
with open(client_csv, 'r', newline='') as f:
    reader = csv.DictReader(f)
    for r in reader:
        ep = int(r['epoch'])
        total_per_epoch[ep] += 1
        if str(r['is_selected']).lower() in ('1', 'true', 'yes'):
            selected_per_epoch[ep] += 1

if not total_per_epoch:
    print('[DIAG] fedcsap_client_metrics.csv has no rows.')
    raise SystemExit(0)

epochs = sorted(total_per_epoch.keys())
ratios = [selected_per_epoch[e] / total_per_epoch[e] for e in epochs]
print(f'[DIAG] selected-client ratio by epoch: min={min(ratios):.3f}, max={max(ratios):.3f}, mean={sum(ratios)/len(ratios):.3f}')
if max(ratios) <= 0.1:
    print('[DIAG][WARN] Very few clients selected by FedCSAP (<=10%); aggregation may be overly conservative.')
PY
else
  echo "[DIAG][WARN] ${CLIENT_CSV} not found; skip FedCSAP selection diagnosis."
fi