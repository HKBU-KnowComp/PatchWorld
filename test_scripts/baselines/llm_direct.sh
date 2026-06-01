#!/usr/bin/env bash
# Run the LLM-Direct baseline on PatchWorld trajectory splits.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/common.sh"

require_abductworld_repo
ensure_api_key

export PYTHONPATH="${ABDUCTWORLD_REPO}:${PYTHONPATH:-}"
export ARTIFACTS_ROOT="${ARTIFACTS_ROOT:-${PATCHWORLD_REPO_ROOT}/artifacts/patchworld}"

MODEL_SLUG="$(baseline_model_slug "${MODEL}")"
MODEL_SLUG="${MODEL_SLUG:-model}"
OUT_DIR="${LLM_DIRECT_OUT_DIR:-${PATCHWORLD_BASELINE_ARTIFACTS_ROOT}/results/llm_direct/${MODEL_SLUG}}"
LOG_DIR="${LLM_DIRECT_LOG_DIR:-${PATCHWORLD_BASELINE_ARTIFACTS_ROOT}/logs/llm_direct/${MODEL_SLUG}}"
RQ3_OUT_DIR="${RQ3_OUT_DIR:-${PATCHWORLD_BASELINE_ARTIFACTS_ROOT}/results/rq3/llm-direct}"
RQ3_LOG_DIR="${RQ3_LOG_DIR:-${PATCHWORLD_BASELINE_ARTIFACTS_ROOT}/logs/rq3}"
MAX_PARALLEL="${MAX_PARALLEL:-1}"

EXPERIMENT="${EXPERIMENT:-all}"
MAX_TRAIN="${MAX_TRAIN:-0}"
MAX_EVAL_OBS="${MAX_EVAL_OBS:-0}"
MAX_INDEX_SIZE="${MAX_INDEX_SIZE:-50000}"
K="${K:-3}"
RQ1_WORKERS="${RQ1_WORKERS:-5}"
RQ2_WORKERS="${RQ2_WORKERS:-5}"
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
  ENVS="${ENVS:-maze}"
  MAX_TRAIN="${MAX_TRAIN:-3}"
  MAX_EVAL_OBS="${MAX_EVAL_OBS:-30}"
  MAX_PARALLEL="${MAX_PARALLEL:-1}"
  AGENT_NUM_TASKS="${AGENT_NUM_TASKS:-3}"
  AGENT_TASK_CAP="${AGENT_TASK_CAP:-3}"
  MAX_PARALLEL_ENVS="${MAX_PARALLEL_ENVS:-1}"
fi

mkdir -p "${OUT_DIR}" "${LOG_DIR}"
split_envs "${ENVS}"

cd "${ABDUCTWORLD_REPO}"

echo "[llm-direct] data=${DATA_ROOT}"
echo "[llm-direct] output=${OUT_DIR}"
echo "[llm-direct] logs=${LOG_DIR}"

run_rq12() {
  pids=()
  job_names=()
  log_files=()

  for env in "${ENV_ARR[@]}"; do
    env="$(echo "${env}" | xargs)"
    [[ -z "${env}" ]] && continue

    while [[ "$(jobs -rp | wc -l)" -ge "${MAX_PARALLEL}" ]]; do
      sleep 1
    done

    log_file="${LOG_DIR}/llm_direct_${env}.log"
    echo "[llm-direct] RQ1/RQ2 start env=${env} log=${log_file}"
    (
      python -u -m test_scripts.abductworld.run_llm_direct_benchmark \
        --experiment "${EXPERIMENT}" \
        --envs "${env}" \
        --split_root "${DATA_ROOT}" \
        --eval_split "${EVAL_SPLIT}" \
        --max_train "${MAX_TRAIN}" \
        --max_eval_obs "${MAX_EVAL_OBS}" \
        --max_index_size "${MAX_INDEX_SIZE}" \
        --k "${K}" \
        --rq1_workers "${RQ1_WORKERS}" \
        --rq2_workers "${RQ2_WORKERS}" \
        --rq2_max_horizon "${RQ2_MAX_HORIZON}" \
        --rq2_report_steps "${RQ2_REPORT_STEPS}" \
        --model "${MODEL}" \
        --output_dir "${OUT_DIR}" \
        "$@" \
        > "${log_file}" 2>&1
    ) &
    pids+=("$!")
    job_names+=("env=${env}")
    log_files+=("${log_file}")
  done

  failed=()
  for i in "${!pids[@]}"; do
    if ! wait "${pids[$i]}"; then
      failed+=("${job_names[$i]} (log: ${log_files[$i]})")
    fi
  done

  if [[ "${#failed[@]}" -gt 0 ]]; then
    echo "[llm-direct] failed jobs:" >&2
    printf '  - %s\n' "${failed[@]}" >&2
    exit 1
  fi

  echo "[llm-direct] RQ1/RQ2 done. Results: ${OUT_DIR}"
}

run_rq3() {
  mkdir -p "${RQ3_OUT_DIR}" "${RQ3_LOG_DIR}"
  local stamp
  stamp="$(date +%Y%m%d_%H%M%S)"
  local log_file="${RQ3_LOG_DIR}/llm-direct_${stamp}.log"
  echo "[llm-direct] RQ3 output=${RQ3_OUT_DIR}"
  echo "[llm-direct] RQ3 log=${log_file}"
  python -u -m abductworld.evaluate_rq3 \
    --run_label "${RQ3_RESULTS_LABEL:-patchworld}" \
    --methods llm-direct \
    --envs "${ENVS}" \
    --model "${MODEL}" \
    --base_url "${BASE_URL}" \
    --artifacts_root "${ARTIFACTS_ROOT}" \
    --split_root "${DATA_ROOT}" \
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
    --llm_direct_k "${K}" \
    --llm_direct_max_index_size "${MAX_INDEX_SIZE}" \
    "$@" \
    2>&1 | tee -a "${log_file}"
}

if rq_wants rq1 || rq_wants rq2; then
  run_rq12 "$@"
fi

if rq_wants rq3; then
  run_rq3 "$@"
fi
