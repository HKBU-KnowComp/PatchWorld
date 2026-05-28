#!/usr/bin/env bash
set -euo pipefail

# Bootstrap AgentGym *server* conda environments for PatchWorld RQ3.
#
# PatchWorld itself: bash scripts/install.sh
#
# Usage:
#   bash scripts/install_agentgym_envs.sh
#   ONLY_ENVS=lmrlgym bash scripts/install_agentgym_envs.sh
#   CONDA_ENV_LMRLGYM=pw-test-lmrlgym ONLY_ENVS=lmrlgym bash scripts/install_agentgym_envs.sh
#
# Conda env name overrides (defaults match AgentGym server package names):
#   CONDA_ENV_ALFWORLD, CONDA_ENV_SCIWORLD, CONDA_ENV_BABYAI,
#   CONDA_ENV_LMRLGYM, CONDA_ENV_TEXTCRAFT, CONDA_ENV_WEBSHOP

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
AGENTGYM_DIR="${AGENTGYM_DIR:-${REPO_ROOT}/../AgentGym}"
ONLY_ENVS="${ONLY_ENVS:-}"

CONDA_ENV_ALFWORLD="${CONDA_ENV_ALFWORLD:-agentenv-alfworld}"
CONDA_ENV_SCIWORLD="${CONDA_ENV_SCIWORLD:-agentenv-sciworld}"
CONDA_ENV_BABYAI="${CONDA_ENV_BABYAI:-agentenv-babyai}"
CONDA_ENV_LMRLGYM="${CONDA_ENV_LMRLGYM:-agentenv-lmrlgym}"
CONDA_ENV_TEXTCRAFT="${CONDA_ENV_TEXTCRAFT:-agentenv-textcraft}"
CONDA_ENV_WEBSHOP="${CONDA_ENV_WEBSHOP:-agentenv-webshop}"

if [[ ! -d "${AGENTGYM_DIR}" ]]; then
  echo "[install-agentgym] AgentGym dir not found: ${AGENTGYM_DIR}" >&2
  echo "[install-agentgym] Clone first: git clone --recursive https://github.com/marcos0318/AgentGym ${AGENTGYM_DIR}" >&2
  exit 1
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "[install-agentgym] conda not found in PATH." >&2
  exit 1
fi

env_selected() {
  local key="$1"
  if [[ -z "${ONLY_ENVS}" ]]; then
    return 0
  fi
  IFS=',' read -r -a wanted <<< "${ONLY_ENVS}"
  for item in "${wanted[@]}"; do
    item="$(echo "${item}" | xargs)"
    if [[ "${item}" == "${key}" ]]; then
      return 0
    fi
  done
  return 1
}

conda_env_exists() {
  conda env list | awk '{print $1}' | grep -qx "$1"
}

ensure_conda_env() {
  local env_name="$1"
  local python_version="$2"
  if conda_env_exists "${env_name}"; then
    echo "[install-agentgym] reusing conda env ${env_name}"
    return 0
  fi
  echo "[install-agentgym] creating conda env ${env_name} (python=${python_version})"
  conda create -y -n "${env_name}" "python=${python_version}"
}

run_in_env() {
  local env_name="$1"
  shift
  conda run -n "${env_name}" bash -lc "$*"
}

install_alfworld() {
  env_selected alfworld || return 0
  local pkg="${AGENTGYM_DIR}/agentenv-alfworld"
  ensure_conda_env "${CONDA_ENV_ALFWORLD}" 3.9
  echo "[install-agentgym] setting up alfworld in ${CONDA_ENV_ALFWORLD}"
  run_in_env "${CONDA_ENV_ALFWORLD}" "cd \"${pkg}\" && bash ./setup.sh"
}

install_sciworld() {
  env_selected sciworld || return 0
  local pkg="${AGENTGYM_DIR}/agentenv-sciworld"
  ensure_conda_env "${CONDA_ENV_SCIWORLD}" 3.8
  echo "[install-agentgym] setting up sciworld in ${CONDA_ENV_SCIWORLD}"
  run_in_env "${CONDA_ENV_SCIWORLD}" "cd \"${pkg}\" && python -m pip install -U pip && python -m pip install -e ."
}

install_babyai() {
  env_selected babyai || return 0
  local pkg="${AGENTGYM_DIR}/agentenv-babyai"
  ensure_conda_env "${CONDA_ENV_BABYAI}" 3.9
  echo "[install-agentgym] setting up babyai in ${CONDA_ENV_BABYAI}"
  run_in_env "${CONDA_ENV_BABYAI}" "cd \"${pkg}\" && python -m pip install -U pip && python -m pip install -e ."
}

install_lmrlgym() {
  env_selected lmrlgym || env_selected maze || env_selected wordle || return 0
  local pkg="${AGENTGYM_DIR}/agentenv-lmrlgym"
  echo "[install-agentgym] initializing lmrlgym submodules"
  git -C "${AGENTGYM_DIR}" submodule update --init --recursive agentenv-lmrlgym/lmrlgym || true
  if ! conda_env_exists "${CONDA_ENV_LMRLGYM}"; then
    echo "[install-agentgym] creating conda env ${CONDA_ENV_LMRLGYM} from environment.yml"
    conda env create -n "${CONDA_ENV_LMRLGYM}" -f "${pkg}/lmrlgym/environment.yml"
  else
    echo "[install-agentgym] reusing conda env ${CONDA_ENV_LMRLGYM}"
  fi
  echo "[install-agentgym] setting up lmrlgym in ${CONDA_ENV_LMRLGYM}"
  run_in_env "${CONDA_ENV_LMRLGYM}" "cd \"${pkg}\" && bash ./setup.sh"
}

install_textcraft() {
  env_selected textcraft || return 0
  local pkg="${AGENTGYM_DIR}/agentenv-textcraft"
  ensure_conda_env "${CONDA_ENV_TEXTCRAFT}" 3.9
  echo "[install-agentgym] setting up textcraft in ${CONDA_ENV_TEXTCRAFT}"
  run_in_env "${CONDA_ENV_TEXTCRAFT}" "cd \"${pkg}\" && python -m pip install -U pip && python -m pip install -e ."
}

install_webshop() {
  env_selected webshop || return 0
  local pkg="${AGENTGYM_DIR}/agentenv-webshop"
  if ! conda_env_exists "${CONDA_ENV_WEBSHOP}"; then
    echo "[install-agentgym] creating conda env ${CONDA_ENV_WEBSHOP} from environment.yml"
    conda env create -n "${CONDA_ENV_WEBSHOP}" -f "${pkg}/environment.yml"
  else
    echo "[install-agentgym] reusing conda env ${CONDA_ENV_WEBSHOP}"
  fi
  echo "[install-agentgym] setting up webshop in ${CONDA_ENV_WEBSHOP}"
  run_in_env "${CONDA_ENV_WEBSHOP}" "cd \"${pkg}\" && bash ./setup.sh"
}

install_alfworld
install_sciworld
install_babyai
install_lmrlgym
install_textcraft
install_webshop

echo "[install-agentgym] done"
