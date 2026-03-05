#!/usr/bin/env bash
set -euo pipefail

# Use a dedicated YAML to avoid jinja/context side effects and test whether
# PathMNIST training can improve under clean FedAvg.

PARAMS="${PARAMS:-utils/pathmnist_clean_fedavg.yaml}"
RUN_TAG="${RUN_TAG:-run_clean_pathmnist_fedavg_params}"

if [[ ! -f "${PARAMS}" ]]; then
  echo "[ERROR] params file not found: ${PARAMS}"
  exit 1
fi

if [[ "${RUN_TAG}" != run_* ]]; then
  echo "[ERROR] RUN_TAG must start with run_"
  exit 1
fi

mkdir -p "runs/${RUN_TAG}"

echo "[INFO] Starting clean PathMNIST FedAvg run from params"
echo "[INFO] PARAMS=${PARAMS}, RUN_TAG=${RUN_TAG}"

cmd=(
  python main.py
  --params="${PARAMS}"
  --"${RUN_TAG}"
)

printf '[CMD]'; printf ' %q' "${cmd[@]}"; printf '\n'
"${cmd[@]}"

GLOBAL_CSV="runs/${RUN_TAG}/global_metrics.csv"
if [[ -f "${GLOBAL_CSV}" ]]; then
  python - <<'PY'
import csv, os
run_tag = os.environ.get('RUN_TAG', 'run_clean_pathmnist_fedavg_params')
global_csv = f'runs/{run_tag}/global_metrics.csv'
rows = []
with open(global_csv, 'r', newline='') as f:
    rows = list(csv.DictReader(f))
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
  echo "[DIAG][WARN] ${GLOBAL_CSV} not found."
fi