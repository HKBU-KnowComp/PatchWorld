#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

SOURCE_ROOT="${SOURCE_ROOT:?Set SOURCE_ROOT to the directory containing per-env split JSONLs}"
OUTPUT_DIR="${OUTPUT_DIR:-artifacts/patchworld/data_release}"
ENVS="${ENVS:-alfworld,babyai,maze,sciworld,textcraft,webshop,wordle}"

python -m patchworld.cli.package_data \
  --source_root "${SOURCE_ROOT}" \
  --output_dir "${OUTPUT_DIR}" \
  --envs "${ENVS}" \
  --archive \
  "$@"
