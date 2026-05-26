#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

ENVS="${ENVS:-alfworld,babyai,maze,sciworld,textcraft,webshop,wordle}"
DATA_ROOT="${DATA_ROOT:-artifacts/patchworld/data_release}"
OUT_DIR="${OUT_DIR:-artifacts/patchworld/results/rq2}"
ROLLOUT_HORIZON="${ROLLOUT_HORIZON:-15}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-1,2,3,5,8,10,15}"

IFS=',' read -r -a ENV_ARR <<< "${ENVS}"
for env in "${ENV_ARR[@]}"; do
  env="$(echo "${env}" | xargs)"
  [[ -z "${env}" ]] && continue
  echo "[run-rq2] env=${env}"
  python -m patchworld.cli.benchmark \
    --env "${env}" \
    --train_glob "${DATA_ROOT}/${env}/${env}_traj_train.jsonl" \
    --eval_glob "${DATA_ROOT}/${env}/${env}_traj_test.jsonl" \
    --output_dir "${OUT_DIR}" \
    --run_rollout \
    --rollout_horizon "${ROLLOUT_HORIZON}" \
    --rollout_steps "${ROLLOUT_STEPS}" \
    "$@"
done
