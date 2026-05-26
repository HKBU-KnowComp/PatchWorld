#!/usr/bin/env bash
set -euo pipefail

# Stop all AgentGym servers on default ports.
# Matches start_agentgym_servers.sh core/full profiles.

ports=(36001 36002 36003 36004 36005 36006 36007 36008 36009 36010 36011 36012 36013 36014)
args=()
for p in "${ports[@]}"; do
  args+=(-ti:"${p}")
done

pids="$(lsof "${args[@]}" 2>/dev/null || true)"
if [[ -z "${pids}" ]]; then
  echo "[stop-agentgym] no processes found on AgentGym ports"
  exit 0
fi

echo "[stop-agentgym] stopping pids: ${pids//$'\n'/ }"
kill ${pids}
echo "[stop-agentgym] done"
