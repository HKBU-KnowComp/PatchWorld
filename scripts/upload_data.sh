#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

HF_REPO="${HF_REPO:-HKBU-KnowComp/patchworld-trajectories}"
SOURCE_DIR="${SOURCE_DIR:-artifacts/patchworld/data_release}"
COMMIT_MESSAGE="${COMMIT_MESSAGE:-Upload PatchWorld trajectory splits}"

if ! command -v hf >/dev/null 2>&1; then
  echo "[upload-data] huggingface_hub CLI not found. Install with: pip install -U huggingface_hub"
  exit 1
fi

if [[ -z "${HF_TOKEN:-}" && -z "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
  echo "[upload-data] set HF_TOKEN to a Hugging Face token with write access"
  echo "[upload-data] create one at: https://huggingface.co/settings/tokens"
fi

if [[ ! -d "${SOURCE_DIR}" ]]; then
  echo "[upload-data] source directory not found: ${SOURCE_DIR}"
  echo "[upload-data] run: bash scripts/package_data.sh"
  exit 1
fi

if [[ ! -f "${SOURCE_DIR}/manifest.json" ]]; then
  echo "[upload-data] missing manifest.json in ${SOURCE_DIR}"
  exit 1
fi

echo "[upload-data] repo=${HF_REPO}"
echo "[upload-data] source=${SOURCE_DIR}"

hf repo create "${HF_REPO}" --type dataset --exist-ok

hf upload "${HF_REPO}" "${SOURCE_DIR}" . \
  --repo-type dataset \
  --exclude "*.tar.gz" \
  --commit-message "${COMMIT_MESSAGE}"

DATASET_README="${REPO_ROOT}/data/huggingface/README.md"
if [[ -f "${DATASET_README}" ]]; then
  hf upload "${HF_REPO}" "${DATASET_README}" README.md \
    --repo-type dataset \
    --commit-message "Add dataset card"
fi

echo "[upload-data] uploaded -> https://huggingface.co/datasets/${HF_REPO}"
