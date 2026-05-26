#!/usr/bin/env bash
set -euo pipefail

# Start AgentGym servers on the standard PatchWorld ports.
#
# Profiles:
#   core: paper environments used by PatchWorld (default)
#     alfworld, sciworld, babyai, lmrlgym(maze/wordle), textcraft, webshop
#   full: add webarena + tool envs + searchqa + sqlgym
#
# Usage:
#   bash scripts/start_agentgym_servers.sh
#   PROFILE=full bash scripts/start_agentgym_servers.sh
#
# Stop:
#   bash scripts/stop_agentgym_servers.sh

PROFILE="${PROFILE:-core}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TEXTCRAFT_DIR="${TEXTCRAFT_DIR:-${REPO_ROOT}/../AgentGym/agentenv-textcraft}"
JAVA_TOOL_OPTIONS="${JAVA_TOOL_OPTIONS:--Xmx96g}"
export JAVA_TOOL_OPTIONS

ulimit -n 65535 || true

pids=()

start_bg() {
  local name="$1"
  shift
  echo "[start-agentgym] starting ${name}: $*"
  "$@" &
  pids+=($!)
}

# Core profile.
start_bg alfworld conda run -n agentenv-alfworld alfworld --host 0.0.0.0 --port 36001
start_bg sciworld conda run -n agentenv-sciworld sciworld --host 0.0.0.0 --port 36002
start_bg babyai conda run -n agentenv-babyai babyai --host 0.0.0.0 --port 36003
start_bg lmrlgym conda run -n agentenv-lmrlgym lmrlgym --host 0.0.0.0 --port 36004
start_bg textcraft bash -lc "cd \"${TEXTCRAFT_DIR}\" && conda run -n agentenv-textcraft textcraft --host 0.0.0.0 --port 36005"
start_bg webshop conda run -n agentenv-webshop webshop --host 0.0.0.0 --port 36006

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

echo "[start-agentgym] started ${#pids[@]} processes; waiting"
wait
