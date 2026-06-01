#!/usr/bin/env bash
# Run the paper Word2World/Qwen-observation baseline training and RQ evals.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/common.sh"

require_abductworld_repo

export PYTHONPATH="${ABDUCTWORLD_REPO}:${PYTHONPATH:-}"
export ARTIFACTS_ROOT="${ARTIFACTS_ROOT:-${PATCHWORLD_REPO_ROOT}/artifacts/patchworld}"

BASE_MODEL="${BASE_MODEL:-${QWEN_OBS_BASE_MODEL:-Qwen/Qwen3.5-4B}}"
MODEL_DIR="${MODEL_DIR:-${QWEN_OBS_MODEL_DIR:-${PATCHWORLD_BASELINE_ARTIFACTS_ROOT}/results/qwen_observation_sft}}"
TRAIN_OUTPUT_DIR="${TRAIN_OUTPUT_DIR:-${MODEL_DIR}}"
TRAIN_LOG_ROOT="${TRAIN_LOG_ROOT:-${PATCHWORLD_BASELINE_ARTIFACTS_ROOT}/logs/qwen_observation_sft}"
RQ12_OUT_DIR="${WORD2WORLD_RQ12_OUT_DIR:-${PATCHWORLD_BASELINE_ARTIFACTS_ROOT}/results/word2world_qwen_observation/rq12}"
RQ12_LOG_DIR="${WORD2WORLD_RQ12_LOG_DIR:-${PATCHWORLD_BASELINE_ARTIFACTS_ROOT}/logs/word2world_qwen_observation/rq12}"
RQ3_OUT_DIR="${RQ3_OUT_DIR:-${PATCHWORLD_BASELINE_ARTIFACTS_ROOT}/results/rq3/word2world}"
RQ3_LOG_DIR="${RQ3_LOG_DIR:-${PATCHWORLD_BASELINE_ARTIFACTS_ROOT}/logs/rq3}"

EXPERIMENT="${EXPERIMENT:-all}"
MAX_TRAIN="${MAX_TRAIN:-0}"
MAX_EVAL_OBS="${MAX_EVAL_OBS:-0}"
RQ2_MAX_HORIZON="${RQ2_MAX_HORIZON:-15}"
RQ2_REPORT_STEPS="${RQ2_REPORT_STEPS:-1,2,3,5,8,10,15}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
MAX_INPUT_CHARS="${MAX_INPUT_CHARS:-6000}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
TORCH_DTYPE="${TORCH_DTYPE:-auto}"
LOAD_IN_4BIT="${LOAD_IN_4BIT:-0}"

if [[ "${BASELINE_RQ:-all}" == "rq1" ]]; then
  EXPERIMENT="rq1"
elif [[ "${BASELINE_RQ:-all}" == "rq2" ]]; then
  EXPERIMENT="rq2"
elif [[ "${BASELINE_RQ:-all}" == "rq12" ]]; then
  EXPERIMENT="all"
fi

if [[ "${SMOKE_TEST:-0}" == "1" ]]; then
  MAX_TRAIN="${MAX_TRAIN:-3}"
  MAX_EVAL_OBS="${MAX_EVAL_OBS:-3}"
  MAX_TRAIN_EXAMPLES="${MAX_TRAIN_EXAMPLES:-16}"
  MAX_TRAIN_EXAMPLES_PER_ENV="${MAX_TRAIN_EXAMPLES_PER_ENV:-16}"
  DRY_RUN_BUILD_DATA="${DRY_RUN_BUILD_DATA:-1}"
  AGENT_NUM_TASKS="${AGENT_NUM_TASKS:-3}"
  AGENT_TASK_CAP="${AGENT_TASK_CAP:-3}"
  MAX_PARALLEL_ENVS="${MAX_PARALLEL_ENVS:-1}"
fi

qwen_quant_arg="--no-load_in_4bit"
if [[ "${LOAD_IN_4BIT}" == "1" ]]; then
  qwen_quant_arg="--load_in_4bit"
fi
qwen_rq3_quant_arg="--no-qwen_obs_load_in_4bit"
if [[ "${LOAD_IN_4BIT}" == "1" ]]; then
  qwen_rq3_quant_arg="--qwen_obs_load_in_4bit"
fi

run_train() {
  cd "${ABDUCTWORLD_REPO}"
  local train_mode="${WORD2WORLD_TRAIN_MODE:-4cards}"
  echo "[word2world] train mode=${train_mode}"
  echo "[word2world] train split=${DATA_ROOT}"
  echo "[word2world] train output=${TRAIN_OUTPUT_DIR}"
  if [[ "${train_mode}" == "4cards" ]]; then
    ARTIFACTS_ROOT="${ARTIFACTS_ROOT}" \
    SPLIT_ROOT="${DATA_ROOT}" \
    OUTPUT_DIR="${TRAIN_OUTPUT_DIR}" \
    LOG_ROOT="${TRAIN_LOG_ROOT}" \
    BASE_MODEL="${BASE_MODEL}" \
    MAX_TRAIN_EXAMPLES="${MAX_TRAIN_EXAMPLES:-0}" \
    MAX_TRAIN_EXAMPLES_PER_ENV="${MAX_TRAIN_EXAMPLES_PER_ENV:-0}" \
    MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-1024}" \
    NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}" \
    PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-8}" \
    GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}" \
    GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-1}" \
    DRY_RUN_BUILD_DATA="${DRY_RUN_BUILD_DATA:-0}" \
      bash test_scripts/qwen_observation/run_train_qwen_observation_4cards.sh "$@"
  else
    ARTIFACTS_ROOT="${ARTIFACTS_ROOT}" \
    SPLIT_ROOT="${DATA_ROOT}" \
    OUTPUT_DIR="${TRAIN_OUTPUT_DIR}" \
    LOG_DIR="${TRAIN_LOG_ROOT}" \
    BASE_MODEL="${BASE_MODEL}" \
    ENVS="${ENVS}" \
    MAX_TRAIN_EXAMPLES="${MAX_TRAIN_EXAMPLES:-0}" \
    MAX_TRAIN_EXAMPLES_PER_ENV="${MAX_TRAIN_EXAMPLES_PER_ENV:-0}" \
    MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-2048}" \
    NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}" \
    PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}" \
    GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}" \
    LEARNING_RATE="${LEARNING_RATE:-2e-4}" \
    GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-1}" \
    DRY_RUN_BUILD_DATA="${DRY_RUN_BUILD_DATA:-0}" \
      bash test_scripts/qwen_observation/run_train_qwen_observation.sh "$@"
  fi
}

run_rq12() {
  mkdir -p "${RQ12_OUT_DIR}" "${RQ12_LOG_DIR}"
  cd "${ABDUCTWORLD_REPO}"
  local stamp
  stamp="$(date +%Y%m%d_%H%M%S)"
  local log_file="${RQ12_LOG_DIR}/word2world_qwen_observation_${stamp}.log"
  echo "[word2world] RQ1/RQ2 model_dir=${MODEL_DIR}"
  echo "[word2world] RQ1/RQ2 output=${RQ12_OUT_DIR}"
  python -u -m test_scripts.qwen_observation.evaluate_qwen_observation_rq12 \
    --experiment "${EXPERIMENT}" \
    --envs "${ENVS}" \
    --model_dir "${MODEL_DIR}" \
    --base_model "${BASE_MODEL}" \
    --split_root "${DATA_ROOT}" \
    --eval_split "${EVAL_SPLIT}" \
    --max_train "${MAX_TRAIN}" \
    --max_eval_obs "${MAX_EVAL_OBS}" \
    --rq2_max_horizon "${RQ2_MAX_HORIZON}" \
    --rq2_report_steps "${RQ2_REPORT_STEPS}" \
    --output_dir "${RQ12_OUT_DIR}" \
    --device_map "${DEVICE_MAP}" \
    --torch_dtype "${TORCH_DTYPE}" \
    "${qwen_quant_arg}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --max_input_chars "${MAX_INPUT_CHARS}" \
    "$@" \
    2>&1 | tee -a "${log_file}"
}

run_rq3() {
  mkdir -p "${RQ3_OUT_DIR}" "${RQ3_LOG_DIR}"
  cd "${ABDUCTWORLD_REPO}"
  local stamp
  stamp="$(date +%Y%m%d_%H%M%S)"
  local log_file="${RQ3_LOG_DIR}/word2world_qwen_observation_${stamp}.log"
  echo "[word2world] RQ3 model_dir=${MODEL_DIR}"
  echo "[word2world] RQ3 output=${RQ3_OUT_DIR}"
  python -u -m abductworld.evaluate_rq3 \
    --run_label "${RQ3_RESULTS_LABEL:-patchworld}" \
    --methods qwen-sft-observation \
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
    --qwen_obs_model_dir "${MODEL_DIR}" \
    --qwen_obs_base_model "${BASE_MODEL}" \
    --qwen_obs_device_map "${DEVICE_MAP}" \
    --qwen_obs_torch_dtype "${TORCH_DTYPE}" \
    --qwen_obs_max_new_tokens "${MAX_NEW_TOKENS}" \
    --qwen_obs_max_input_chars "${MAX_INPUT_CHARS}" \
    "${qwen_rq3_quant_arg}" \
    "$@" \
    2>&1 | tee -a "${log_file}"
}

echo "[word2world] using abduct-world qwen_observation scripts"
echo "[word2world] data=${DATA_ROOT}"

if rq_wants train; then
  run_train "$@"
fi

if rq_wants rq1 || rq_wants rq2; then
  run_rq12 "$@"
fi

if rq_wants rq3; then
  ensure_api_key
  run_rq3 "$@"
fi
