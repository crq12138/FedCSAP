#!/usr/bin/env bash
set -euo pipefail

# 运行一次「干净 PathMNIST + FedAvg」实验，并在结束后输出简要诊断。
# 用于和 FedCSAP 对照：
# - 若 FedAvg 也几乎不学习，通常是配置/数据/训练管线问题；
# - 若 FedAvg 能学习而 FedCSAP 不能，通常是 FedCSAP 选择/聚合策略问题。

EPOCHS="${EPOCHS:-50}"
# 与 `test_fl.fl_system.main` 默认配置对齐：20 客户端、每轮全参与。
TOTAL_PARTICIPANTS="${TOTAL_PARTICIPANTS:-25}"
NO_MODELS="${NO_MODELS:-20}"
COMMITTEE_SIZE="${COMMITTEE_SIZE:-5}"
LR_ETA="${LR_ETA:-1.0}"
LR="${LR:-0.1}"
NONIID_MODE="${NONIID_MODE:-sampling_dirichlet}"
INTERNAL_EPOCHS="${INTERNAL_EPOCHS:-2}"
SEED="${SEED:-0}"
RUN_TAG="${RUN_TAG:-run_clean_pathmnist_fedavg}"

if [[ "${RUN_TAG}" != run_* ]]; then
  echo "[ERROR] RUN_TAG 必须以 run_ 开头，例如 run_clean_pathmnist_fedavg"
  exit 1
fi

mkdir -p "runs/${RUN_TAG}"

echo "[INFO] Starting clean PathMNIST FedAvg run"
echo "[INFO] RUN_TAG=${RUN_TAG}, epochs=${EPOCHS}, participants=${TOTAL_PARTICIPANTS}, committee=${COMMITTEE_SIZE}, no_models=${NO_MODELS}, lr=${LR}, eta=${LR_ETA}, noniid=${NONIID_MODE}, internal_epochs=${INTERNAL_EPOCHS}"

cmd=(
  python main.py
  --type=pathmnist
  --aggregation_methods=fedavg
  --attack_methods=targeted_label_flip
  --is_poison=false
  --mal_pcnt=0
  --number_of_adversary_targeted_label_flip=0
  --resumed_model=false
  --epochs="${EPOCHS}"
  --lr="${LR}"
  --number_of_total_participants="${TOTAL_PARTICIPANTS}"
  --no_models="${NO_MODELS}"
  --committee_size="${COMMITTEE_SIZE}"
  --noniid="${NONIID_MODE}"
  --eta="${LR_ETA}"
  --internal_epochs="${INTERNAL_EPOCHS}"
  --minimize_logging=false
  --seed="${SEED}"
  --"${RUN_TAG}"
)

printf '[CMD]'; printf ' %q' "${cmd[@]}"; printf '\n'
"${cmd[@]}"

echo "[INFO] Run finished. Check outputs under runs/${RUN_TAG}/"

GLOBAL_CSV="runs/${RUN_TAG}/global_metrics.csv"
if [[ -f "${GLOBAL_CSV}" ]]; then
  python - <<'PY'
import csv, os
run_tag = os.environ.get('RUN_TAG', 'run_clean_pathmnist_fedavg')
global_csv = f'runs/{run_tag}/global_metrics.csv'
rows = []
with open(global_csv, 'r', newline='') as f:
    reader = csv.DictReader(f)
    rows.extend(reader)
if not rows:
    print('[DIAG] global_metrics.csv is empty.')
    raise SystemExit(0)
start_acc = float(rows[0]['global_acc'])
end_acc = float(rows[-1]['global_acc'])
start_f1 = float(rows[0]['global_macro_f1'])
end_f1 = float(rows[-1]['global_macro_f1'])
print(f'[DIAG] global_acc: start={start_acc:.4f}, end={end_acc:.4f}, delta={end_acc-start_acc:.4f}')
print(f'[DIAG] global_macro_f1: start={start_f1:.4f}, end={end_f1:.4f}, delta={end_f1-start_f1:.4f}')
if abs(end_acc-start_acc) < 0.5:
    print('[DIAG][WARN] Accuracy change < 0.5, learning appears stalled.')
PY
else
  echo "[DIAG][WARN] ${GLOBAL_CSV} not found; skip global metric diagnosis."
fi
