#!/usr/bin/env bash
# Run paper-style PoE-World baselines: offline RQ1/RQ2 and shared-planner RQ3.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/common.sh"

require_abductworld_repo
warn_poe_world_revision
ensure_api_key
warn_if_not_conda_env "${POEWORLD_CONDA_ENV:-poeworld}"

export PYTHONPATH="${ABDUCTWORLD_REPO}:${PYTHONPATH:-}"
export ARTIFACTS_ROOT="${ARTIFACTS_ROOT:-${PATCHWORLD_REPO_ROOT}/artifacts/patchworld}"

PROGRAMMATIC_ROOT="${PROGRAMMATIC_ROOT:-${PATCHWORLD_BASELINE_ARTIFACTS_ROOT}/results/programmatic_rq12}"
POEWORLD_OUT_DIR="${POEWORLD_OUT_DIR:-${PROGRAMMATIC_ROOT}/poe_world}"
RQ3_OUT_DIR="${RQ3_OUT_DIR:-${PATCHWORLD_BASELINE_ARTIFACTS_ROOT}/results/rq3/poe-world}"
RQ3_LOG_DIR="${RQ3_LOG_DIR:-${PATCHWORLD_BASELINE_ARTIFACTS_ROOT}/logs/rq3}"

EXPERIMENT="${EXPERIMENT:-all}"
MAX_TRAIN="${MAX_TRAIN:-0}"
MAX_EVAL_OBS="${MAX_EVAL_OBS:-0}"
MAX_TRAIN_TRANSITIONS="${MAX_TRAIN_TRANSITIONS:-0}"
MAX_BUDGET="${MAX_BUDGET:-15.0}"
MAX_SYNTHESIS_ROUNDS="${MAX_SYNTHESIS_ROUNDS:-15}"
BANDITS_C="${BANDITS_C:-25.0}"
RQ2_MAX_HORIZON="${RQ2_MAX_HORIZON:-15}"
RQ2_REPORT_STEPS="${RQ2_REPORT_STEPS:-1,2,3,5,8,10,15}"

if [[ "${BASELINE_RQ:-all}" == "rq1" ]]; then
  EXPERIMENT="rq1"
elif [[ "${BASELINE_RQ:-all}" == "rq2" ]]; then
  EXPERIMENT="rq2"
elif [[ "${BASELINE_RQ:-all}" == "rq12" ]]; then
  EXPERIMENT="all"
fi

if [[ "${SMOKE_TEST:-0}" == "1" ]]; then
  MAX_TRAIN="${MAX_TRAIN:-3}"
  MAX_EVAL_OBS="${MAX_EVAL_OBS:-30}"
  MAX_TRAIN_TRANSITIONS="${MAX_TRAIN_TRANSITIONS:-200}"
  AGENT_NUM_TASKS="${AGENT_NUM_TASKS:-3}"
  AGENT_TASK_CAP="${AGENT_TASK_CAP:-3}"
  MAX_PARALLEL_ENVS="${MAX_PARALLEL_ENVS:-1}"
fi

run_rq12() {
  mkdir -p "${POEWORLD_OUT_DIR}"
  cd "${ABDUCTWORLD_REPO}"
  echo "[poeworld] RQ1/RQ2 output=${POEWORLD_OUT_DIR}"
  python -u -m test_scripts.programmatic.run_programmatic_rq12 \
    --method poe_world \
    --experiment "${EXPERIMENT}" \
    --envs "${ENVS}" \
    --api_key "${DEEPINFRA_API_KEY}" \
    --base_url "${BASE_URL}" \
    --model "${MODEL}" \
    --poe_world_repo "${POE_WORLD_REPO}" \
    --split_root "${DATA_ROOT}" \
    --eval_split "${EVAL_SPLIT}" \
    --max_train "${MAX_TRAIN}" \
    --max_eval_obs "${MAX_EVAL_OBS}" \
    --max_train_transitions "${MAX_TRAIN_TRANSITIONS}" \
    --max_budget "${MAX_BUDGET}" \
    --max_synthesis_rounds "${MAX_SYNTHESIS_ROUNDS}" \
    --bandits_c "${BANDITS_C}" \
    --rq2_max_horizon "${RQ2_MAX_HORIZON}" \
    --rq2_report_steps "${RQ2_REPORT_STEPS}" \
    --output_dir "${POEWORLD_OUT_DIR}" \
    "$@"
}

run_rq3() {
  mkdir -p "${RQ3_OUT_DIR}" "${RQ3_LOG_DIR}"
  cd "${ABDUCTWORLD_REPO}"
  local stamp
  stamp="$(date +%Y%m%d_%H%M%S)"
  local log_file="${RQ3_LOG_DIR}/poe-world_${stamp}.log"
  echo "[poeworld] RQ3 output=${RQ3_OUT_DIR}"
  echo "[poeworld] RQ3 log=${log_file}"
  python -u -m abductworld.evaluate_rq3 \
    --run_label "${RQ3_RESULTS_LABEL:-patchworld}" \
    --methods poe-world \
    --envs "${ENVS}" \
    --model "${MODEL}" \
    --base_url "${BASE_URL}" \
    --artifacts_root "${ARTIFACTS_ROOT}" \
    --split_root "${DATA_ROOT}" \
    --unified_world_models_root "${PROGRAMMATIC_ROOT}" \
    --output_dir "${RQ3_OUT_DIR}" \
    --eval_split "${EVAL_SPLIT}" \
    --max_parallel_envs "${MAX_PARALLEL_ENVS:-7}" \
    --agent_task_start "${AGENT_TASK_START:-0}" \
    --agent_num_tasks "${AGENT_NUM_TASKS:-200}" \
    --agent_task_cap "${AGENT_TASK_CAP:-200}" \
    --use_split_task_indices \
    --no-artifact_require_models \
    --max_steps "${MAX_STEPS:-30}" \
    --llm_candidate_actions "${LLM_CANDIDATE_ACTIONS:-5}" \
    --wm_policy "${WM_POLICY:-gated-rerank}" \
    --seed "${SEED:-0}" \
    "$@" \
    2>&1 | tee -a "${log_file}"
}

echo "[poeworld] data=${DATA_ROOT}"
echo "[poeworld] poe_world_repo=${POE_WORLD_REPO}"

if rq_wants rq1 || rq_wants rq2; then
  run_rq12 "$@"
fi

if rq_wants rq3; then
  run_rq3 "$@"
fi
