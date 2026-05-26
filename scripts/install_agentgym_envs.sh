#!/usr/bin/env bash
set -euo pipefail

# Install AgentGym source packages for PatchWorld RQ3/live-agent runs.
#
# Behavior:
#  1) Install core `agentenv` first.
#  2) Install each `agentenv-*` environment package from source.
#
# Usage examples:
#   bash scripts/install_agentgym_envs.sh
#   AGENTGYM_DIR=/path/to/AgentGym bash scripts/install_agentgym_envs.sh
#   INSTALL_NO_DEPS=1 bash scripts/install_agentgym_envs.sh
#
# Notes:
# - By default this script expects environment-specific conda env names:
#     agentenv-alfworld, agentenv-sciworld, agentenv-babyai, agentenv-lmrlgym,
#     agentenv-textcraft, agentenv-webshop, agentenv-webarena, agentenv-tool,
#     agentenv-searchqa, agentenv-sqlgym.
# - It runs `pip install -e` inside each env via `conda run -n ...`.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
AGENTGYM_DIR="${AGENTGYM_DIR:-${REPO_ROOT}/../AgentGym}"
INSTALL_NO_DEPS="${INSTALL_NO_DEPS:-1}"

if [[ ! -d "${AGENTGYM_DIR}" ]]; then
  echo "[install-agentgym] AgentGym dir not found: ${AGENTGYM_DIR}" >&2
  echo "[install-agentgym] Clone first: git clone --recursive https://github.com/marcos0318/AgentGym ${AGENTGYM_DIR}" >&2
  exit 1
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "[install-agentgym] conda not found in PATH." >&2
  exit 1
fi

PIP_FLAGS=(-e)
if [[ "${INSTALL_NO_DEPS}" == "1" ]]; then
  PIP_FLAGS+=(--no-deps)
fi

run_install() {
  local env_name="$1"
  local pkg_dir="$2"
  if [[ ! -d "${pkg_dir}" ]]; then
    echo "[install-agentgym] skip missing directory: ${pkg_dir}"
    return 0
  fi
  echo "[install-agentgym] installing ${pkg_dir} into conda env ${env_name}"
  conda run -n "${env_name}" python -m pip install "${PIP_FLAGS[@]}" "${pkg_dir}"
}

# Core package first (as requested).
run_install "agentenv-alfworld" "${AGENTGYM_DIR}/agentenv"

# Environment packages.
run_install "agentenv-alfworld" "${AGENTGYM_DIR}/agentenv-alfworld"
run_install "agentenv-sciworld" "${AGENTGYM_DIR}/agentenv-sciworld"
run_install "agentenv-babyai" "${AGENTGYM_DIR}/agentenv-babyai"
run_install "agentenv-lmrlgym" "${AGENTGYM_DIR}/agentenv-lmrlgym"
run_install "agentenv-textcraft" "${AGENTGYM_DIR}/agentenv-textcraft"
run_install "agentenv-webshop" "${AGENTGYM_DIR}/agentenv-webshop"
run_install "agentenv-webarena" "${AGENTGYM_DIR}/agentenv-webarena"
run_install "agentenv-tool" "${AGENTGYM_DIR}/agentenv-tool"
run_install "agentenv-searchqa" "${AGENTGYM_DIR}/agentenv-searchqa"
run_install "agentenv-sqlgym" "${AGENTGYM_DIR}/agentenv-sqlgym"

echo "[install-agentgym] done"
