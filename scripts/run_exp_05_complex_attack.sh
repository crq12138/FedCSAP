#!/usr/bin/env bash
set -euo pipefail

# 一键运行：复杂攻击场景（8恶意方=TLF/SF/IPM/DBA 各2）
# 默认跑完整 21 组：run_373 ~ run_393
#
# 用法：
#   bash scripts/run_exp_05_complex_attack.sh
#   bash scripts/run_exp_05_complex_attack.sh run_380 run_386
#   DRY_RUN=1 bash scripts/run_exp_05_complex_attack.sh

START_RUN=${1:-run_373}
END_RUN=${2:-run_393}

export MAX_PARALLEL=${MAX_PARALLEL:-3}
export DBA_POISONING_PER_BATCH=${DBA_POISONING_PER_BATCH:-60}
export CONFIG_FILE=${CONFIG_FILE:-scripts/config/exp_05_complex_attack/runs.csv}

bash scripts/exp_05_complex_attack.sh "${START_RUN}" "${END_RUN}"
