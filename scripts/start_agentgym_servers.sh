#!/usr/bin/env bash
set -euo pipefail

# Start AgentGym servers on the standard PatchWorld ports.
#
# Usage:
#   bash scripts/start_agentgym_servers.sh
#   ONLY_SERVERS=lmrlgym bash scripts/start_agentgym_servers.sh
#   CONDA_ENV_LMRLGYM=pw-test-lmrlgym ONLY_SERVERS=lmrlgym bash scripts/start_agentgym_servers.sh
#
# Conda env name overrides (defaults match AgentGym server package names):
#   CONDA_ENV_ALFWORLD, CONDA_ENV_SCIWORLD, CONDA_ENV_BABYAI,
#   CONDA_ENV_LMRLGYM, CONDA_ENV_TEXTCRAFT, CONDA_ENV_WEBSHOP

PROFILE="${PROFILE:-core}"
ONLY_SERVERS="${ONLY_SERVERS:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TEXTCRAFT_DIR="${TEXTCRAFT_DIR:-${REPO_ROOT}/../AgentGym/agentenv-textcraft}"
JAVA_TOOL_OPTIONS="${JAVA_TOOL_OPTIONS:--Xmx96g}"
export JAVA_TOOL_OPTIONS

CONDA_ENV_ALFWORLD="${CONDA_ENV_ALFWORLD:-agentenv-alfworld}"
CONDA_ENV_SCIWORLD="${CONDA_ENV_SCIWORLD:-agentenv-sciworld}"
CONDA_ENV_BABYAI="${CONDA_ENV_BABYAI:-agentenv-babyai}"
CONDA_ENV_LMRLGYM="${CONDA_ENV_LMRLGYM:-agentenv-lmrlgym}"
CONDA_ENV_TEXTCRAFT="${CONDA_ENV_TEXTCRAFT:-agentenv-textcraft}"
CONDA_ENV_WEBSHOP="${CONDA_ENV_WEBSHOP:-agentenv-webshop}"

ulimit -n 65535 || true

pids=()

server_selected() {
  local key="$1"
  if [[ -z "${ONLY_SERVERS}" ]]; then
    return 0
  fi
  IFS=',' read -r -a wanted <<< "${ONLY_SERVERS}"
  for item in "${wanted[@]}"; do
    item="$(echo "${item}" | xargs)"
    if [[ "${item}" == "${key}" ]]; then
      return 0
    fi
  done
  return 1
}

start_bg() {
  local name="$1"
  shift
  echo "[start-agentgym] starting ${name}: $*"
  "$@" &
  pids+=($!)
}

if server_selected alfworld; then
  start_bg alfworld conda run -n "${CONDA_ENV_ALFWORLD}" alfworld --host 0.0.0.0 --port 36001
fi
if server_selected sciworld; then
  start_bg sciworld conda run -n "${CONDA_ENV_SCIWORLD}" sciworld --host 0.0.0.0 --port 36002
fi
if server_selected babyai; then
  start_bg babyai conda run -n "${CONDA_ENV_BABYAI}" babyai --host 0.0.0.0 --port 36003
fi
if server_selected lmrlgym; then
  start_bg lmrlgym conda run -n "${CONDA_ENV_LMRLGYM}" lmrlgym --host 0.0.0.0 --port 36004
fi
if server_selected textcraft; then
  start_bg textcraft bash -lc "cd \"${TEXTCRAFT_DIR}\" && conda run -n \"${CONDA_ENV_TEXTCRAFT}\" textcraft --host 0.0.0.0 --port 36005"
fi
if server_selected webshop; then
  start_bg webshop conda run -n "${CONDA_ENV_WEBSHOP}" webshop --host 0.0.0.0 --port 36006
fi

if [[ "${PROFILE}" == "full" ]]; then
  start_bg webarena conda run -n agentenv-webarena webarena --host 0.0.0.0 --port 36007
  start_bg weather conda run -n agentenv-tool weather --host 0.0.0.0 --port 36008
  start_bg todo conda run -n agentenv-tool todo --host 0.0.0.0 --port 36009
  start_bg movie conda run -n agentenv-tool movie --host 0.0.0.0 --port 36010
  start_bg sheet conda run -n agentenv-tool sheet --host 0.0.0.0 --port 36011
  start_bg academia conda run -n agentenv-tool academia --host 0.0.0.0 --port 36012
  start_bg searchqa conda run -n agentenv-searchqa searchqa --host 0.0.0.0 --port 36013
  start_bg sqlgym conda run -n agentenv-sqlgym sqlgym --host 0.0.0.0 --port 36014
fi

if [[ ${#pids[@]} -eq 0 ]]; then
  echo "[start-agentgym] no servers selected" >&2
  exit 1
fi

echo "[start-agentgym] started ${#pids[@]} processes; waiting"
wait
