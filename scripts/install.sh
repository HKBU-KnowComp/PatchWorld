#!/usr/bin/env bash
set -euo pipefail

# Install PatchWorld and the AgentGym client.
#
# Usage (existing venv/conda env):
#   conda activate my-patchworld-env
#   bash scripts/install.sh
#
# Usage (create/use a named conda env):
#   PATCHWORLD_CONDA_ENV=patchworld-mdtest bash scripts/install.sh
#
# Overrides:
#   AGENTGYM_GIT_URL=git+https://github.com/marcos0318/AgentGym.git#subdirectory=agentenv

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
AGENTGYM_GIT_URL="${AGENTGYM_GIT_URL:-git+https://github.com/marcos0318/AgentGym.git#subdirectory=agentenv}"
PATCHWORLD_CONDA_ENV="${PATCHWORLD_CONDA_ENV:-}"
PATCHWORLD_PYTHON="${PATCHWORLD_PYTHON:-3.12}"

run_install() {
  python -m pip install -U pip
  python -m pip install -e .
  python -m pip install "${AGENTGYM_GIT_URL}" --no-deps
  python -m nltk.downloader punkt
  python - <<'PY'
from agentenv.envs import MazeTask
import patchworld
print("[install] patchworld + agentenv import check OK")
PY
}

if [[ -n "${PATCHWORLD_CONDA_ENV}" ]]; then
  if ! command -v conda >/dev/null 2>&1; then
    echo "[install] PATCHWORLD_CONDA_ENV set but conda not found." >&2
    exit 1
  fi
  if ! conda env list | awk '{print $1}' | grep -qx "${PATCHWORLD_CONDA_ENV}"; then
    echo "[install] creating conda env ${PATCHWORLD_CONDA_ENV} (python=${PATCHWORLD_PYTHON})"
    conda create -y -n "${PATCHWORLD_CONDA_ENV}" "python=${PATCHWORLD_PYTHON}"
  fi
  conda run -n "${PATCHWORLD_CONDA_ENV}" bash -lc "
    set -euo pipefail
    cd \"${REPO_ROOT}\"
    python -m pip install -U pip
    python -m pip install -e .
    python -m pip install \"${AGENTGYM_GIT_URL}\" --no-deps
    python -m nltk.downloader punkt
    python - <<'PY'
from agentenv.envs import MazeTask
import patchworld
print('[install] patchworld + agentenv import check OK')
PY
  "
  echo "[install] done (conda env: ${PATCHWORLD_CONDA_ENV})"
  echo "[install] activate with: conda activate ${PATCHWORLD_CONDA_ENV}"
  exit 0
fi

cd "${REPO_ROOT}"
run_install
echo "[install] done"
