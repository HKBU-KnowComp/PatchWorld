#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

HF_REPO="${HF_REPO:-HKBU-KnowComp/patchworld-trajectories}"
OUTPUT_DIR="${OUTPUT_DIR:-artifacts/patchworld/data_release}"

if ! command -v hf >/dev/null 2>&1; then
  echo "[download-data] huggingface_hub CLI not found. Install with: pip install -U huggingface_hub"
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

echo "[download-data] repo=${HF_REPO}"
echo "[download-data] output=${OUTPUT_DIR}"

hf download "${HF_REPO}" \
  --repo-type dataset \
  --local-dir "${OUTPUT_DIR}" \
  --exclude "*.tar.gz" \
  "$@"

echo "[download-data] done -> ${OUTPUT_DIR}"
echo "[download-data] expected layout: ${OUTPUT_DIR}/<env>/<env>_traj_{train,val,test}.jsonl"
