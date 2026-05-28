---
license: mit
task_categories:
  - other
language:
  - en
tags:
  - agent
  - world-model
  - trajectories
  - agentgym
  - patchworld
size_categories:
  - 10K<n<100K
---

# PatchWorld Trajectory Splits

Trajectory data for reproducing PatchWorld paper experiments (RQ1/RQ2 offline induction and evaluation).

**Code:** [HKBU-KnowComp/PatchWorld](https://github.com/HKBU-KnowComp/PatchWorld)

These files are not hosted in the GitHub repository because of size limits. Download them from this Hugging Face dataset instead.

## Contents

Seven AgentGym environments, each with train/val/test JSONL splits:

- `alfworld/`
- `babyai/`
- `maze/`
- `sciworld/`
- `textcraft/`
- `webshop/`
- `wordle/`

Each directory contains:

- `<env>_traj_train.jsonl`
- `<env>_traj_val.jsonl`
- `<env>_traj_test.jsonl`

`manifest.json` lists file checksums and trajectory counts.

## JSONL Schema

Each line is one trajectory with:

- `metadata`: `env`, `item_id`, `task_idx`, `rollout_index`, `split`, `success`, `total_reward`
- `transitions`: list of `{observation, action, next_observation, reward, done}`

## Download

From the PatchWorld repo root:

```bash
bash scripts/download_data.sh
```

Or with the Hugging Face CLI directly:

```bash
hf download HKBU-KnowComp/patchworld-trajectories \
  --repo-type dataset \
  --local-dir artifacts/patchworld/data_release \
  --exclude "*.tar.gz"
```

Override the destination:

```bash
OUTPUT_DIR=/path/to/data bash scripts/download_data.sh
```

## Usage

After download, run experiments with:

```bash
DATA_ROOT=artifacts/patchworld/data_release bash scripts/run_rq1.sh
```

See the [PatchWorld README](https://github.com/HKBU-KnowComp/PatchWorld) and `EXPERIMENTS.md` for full instructions.
