#!/usr/bin/env bash

# Shared defaults for PatchWorld comparative baseline launchers.

PATCHWORLD_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

ABDUCTWORLD_REPO="${ABDUCTWORLD_REPO:-/data/jbai/abduct-world}"
POE_WORLD_REPO="${POE_WORLD_REPO:-/data/jbai/poe-world}"
POE_WORLD_REPO_URL="${POE_WORLD_REPO_URL:-https://github.com/marcos0318/poe-world.git}"

DATA_ROOT="${DATA_ROOT:-${PATCHWORLD_REPO_ROOT}/artifacts/patchworld/data_release}"
PATCHWORLD_BASELINE_ARTIFACTS_ROOT="${PATCHWORLD_BASELINE_ARTIFACTS_ROOT:-${PATCHWORLD_REPO_ROOT}/artifacts/patchworld/baselines}"

ENVS="${ENVS:-alfworld,babyai,maze,sciworld,textcraft,webshop,wordle}"
MODEL="${MODEL:-Qwen/Qwen3-Coder-480B-A35B-Instruct-Turbo}"
BASE_URL="${BASE_URL:-https://api.deepinfra.com/v1/openai}"
EVAL_SPLIT="${EVAL_SPLIT:-test}"

export POE_WORLD_REPO

baseline_model_slug() {
  printf '%s' "$1" | sed 's/[^A-Za-z0-9._-]/_/g; s/^[._-]*//; s/[._-]*$//'
}

require_dir() {
  local path="$1"
  local label="$2"
  if [[ ! -d "${path}" ]]; then
    echo "[patchworld-baselines] missing ${label}: ${path}" >&2
    exit 1
  fi
}

require_abductworld_repo() {
  require_dir "${ABDUCTWORLD_REPO}" "abduct-world repo"
}

require_poe_world_repo() {
  require_dir "${POE_WORLD_REPO}" "poe-world repo"
}

warn_poe_world_revision() {
  require_poe_world_repo
  local remote_url=""
  if command -v git >/dev/null 2>&1; then
    remote_url="$(git -C "${POE_WORLD_REPO}" remote get-url origin 2>/dev/null || true)"
  fi
  if [[ -n "${remote_url}" && "${remote_url}" != *"marcos0318/poe-world"* ]]; then
    echo "[patchworld-baselines] warning: POE_WORLD_REPO origin is '${remote_url}'." >&2
    echo "[patchworld-baselines] expected revised implementation: ${POE_WORLD_REPO_URL}" >&2
  fi
  echo "[patchworld-baselines] using POE_WORLD_REPO=${POE_WORLD_REPO}" >&2
}

ensure_api_key() {
  if [[ -n "${DEEPINFRA_API_KEY:-}" ]]; then
    export DEEPINFRA_API_KEY
    return
  fi
  if [[ -n "${PATCHWORLD_LLM_API_KEY:-}" ]]; then
    export DEEPINFRA_API_KEY="${PATCHWORLD_LLM_API_KEY}"
    return
  fi

  local candidates=(
    "${PATCHWORLD_REPO_ROOT}/.deepinfra_api_key"
    "${ABDUCTWORLD_REPO}/.deepinfra_api_key"
    "${HOME}/.deepinfra_api_key"
  )
  local path
  for path in "${candidates[@]}"; do
    if [[ -f "${path}" ]]; then
      DEEPINFRA_API_KEY="$(tr -d '\r\n' < "${path}")"
      export DEEPINFRA_API_KEY
      return
    fi
  done

  echo "[patchworld-baselines] missing API key; set DEEPINFRA_API_KEY or PATCHWORLD_LLM_API_KEY." >&2
  exit 1
}

warn_if_not_conda_env() {
  local expected="$1"
  local current="${CONDA_DEFAULT_ENV:-}"
  if [[ -n "${expected}" && "${current}" != "${expected}" ]]; then
    echo "[patchworld-baselines] note: this baseline is expected to run in conda env '${expected}' (current: '${current:-none}')." >&2
  fi
}

split_envs() {
  local raw="$1"
  IFS=',' read -r -a ENV_ARR <<< "${raw}"
}

rq_wants() {
  local target="$1"
  local requested="${BASELINE_RQ:-all}"
  case "${requested}" in
    all) return 0 ;;
    train) [[ "${target}" == "train" ]] ;;
    rq12) [[ "${target}" == "rq1" || "${target}" == "rq2" ]] ;;
    "${target}") return 0 ;;
    *) return 1 ;;
  esac
}
