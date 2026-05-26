#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

ENVS="${ENVS:-alfworld,babyai,maze,sciworld,textcraft,webshop,wordle}"
RQ1_DIR="${RQ1_DIR:-artifacts/patchworld/results/rq1}"
OUT_DIR="${OUT_DIR:-artifacts/patchworld/results/rq3}"
NUM_TASKS="${NUM_TASKS:-200}"
TASK_START="${TASK_START:-0}"
MAX_STEPS="${MAX_STEPS:-30}"

latest_model_for_env() {
  local env="$1"
  python - "$RQ1_DIR" "$env" <<'PY'
from pathlib import Path
import sys
root = Path(sys.argv[1])
env = sys.argv[2]
model_dir = root / "generated_models"
if not model_dir.exists():
    print("")
    raise SystemExit(0)
candidates = sorted(model_dir.glob(f"{env}_patchworld_*.py"), key=lambda p: p.stat().st_mtime, reverse=True)
print(str(candidates[0]) if candidates else "")
PY
}

IFS=',' read -r -a ENV_ARR <<< "${ENVS}"
for env in "${ENV_ARR[@]}"; do
  env="$(echo "${env}" | xargs)"
  [[ -z "${env}" ]] && continue
  model_path="$(latest_model_for_env "${env}")"
  if [[ -z "${model_path}" ]]; then
    echo "[run-rq3] no model found for env=${env} under ${RQ1_DIR}/generated_models"
    echo "[run-rq3] run scripts/run_rq1.sh first or set RQ1_DIR."
    exit 1
  fi
  echo "[run-rq3] env=${env} model=${model_path}"
  python -m patchworld.cli.rq3_agent_eval \
    --env "${env}" \
    --model_path "${model_path}" \
    --num_tasks "${NUM_TASKS}" \
    --task_start "${TASK_START}" \
    --max_steps "${MAX_STEPS}" \
    --output_json "${OUT_DIR}/rq3_${env}.json" \
    "$@"
done
