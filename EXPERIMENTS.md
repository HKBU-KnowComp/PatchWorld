# PatchWorld Experiments Runbook

This document describes how to package data and run PatchWorld experiments
end-to-end in the standalone repo.

## 1) Setup

```bash
cd /path/to/patchworld
python -m venv .venv
source .venv/bin/activate
pip install -e .
python -m nltk.downloader punkt
```

Set API key for induction/evaluation:

```bash
export PATCHWORLD_LLM_API_KEY=...
# or use DEEPINFRA_API_KEY
```

## 2) Package the Data

If you already have split JSONLs in the source tree, package them for this repo:

```bash
bash scripts/package_data.sh
```

Overrides:

```bash
SOURCE_ROOT=/path/to/resplit_train_val_test_seed42 \
OUTPUT_DIR=artifacts/patchworld/data_release \
ENVS=alfworld,babyai,maze,sciworld,textcraft,webshop,wordle \
bash scripts/package_data.sh
```

Output:
- `artifacts/patchworld/data_release/<env>/<env>_traj_{train,val,test}.jsonl`
- `artifacts/patchworld/data_release/manifest.json`
- `artifacts/patchworld/data_release_<timestamp>.tar.gz`

## 3) Run RQ1 (One-step Fidelity)

```bash
bash scripts/run_rq1.sh
```

Common overrides:

```bash
ENVS=maze,wordle \
DATA_ROOT=artifacts/patchworld/data_release \
OUT_DIR=artifacts/patchworld/results/rq1 \
bash scripts/run_rq1.sh --model Qwen/Qwen3-Coder-480B-A35B-Instruct-Turbo
```

Results:
- JSON reports in `artifacts/patchworld/results/rq1/`
- generated models in `artifacts/patchworld/results/rq1/generated_models/`

## 4) Run RQ2 (Rollout Robustness)

```bash
bash scripts/run_rq2.sh
```

Common overrides:

```bash
ROLLOUT_HORIZON=15 \
ROLLOUT_STEPS=1,2,3,5,8,10,15 \
bash scripts/run_rq2.sh
```

Results:
- JSON reports in `artifacts/patchworld/results/rq2/` with `rq2_rollout` blocks.

## 5) Run RQ3 (Live Planning)

### 5.1 Install AgentGym source packages

Use separate conda environments per server package to avoid dependency conflicts
between `agentenv-*` environments.

```bash
bash scripts/install_agentgym_envs.sh
```

Each server package is installed standalone in its own env (`agentenv-alfworld`,
`agentenv-sciworld`, `agentenv-babyai`, `agentenv-lmrlgym`,
`agentenv-textcraft`, `agentenv-webshop`).

For the PatchWorld runner process, install the AgentGym client package once:

```bash
pip install -e ../AgentGym/agentenv --no-deps
```

### 5.2 Start environment servers

```bash
bash scripts/start_agentgym_servers.sh
```

### 5.3 Run RQ3

RQ3 uses the latest per-env induced models from RQ1 output by default:

```bash
bash scripts/run_rq3.sh
```

Overrides:

```bash
RQ1_DIR=artifacts/patchworld/results/rq1 \
OUT_DIR=artifacts/patchworld/results/rq3 \
NUM_TASKS=200 \
MAX_STEPS=30 \
bash scripts/run_rq3.sh --react_baseline
```

Stop servers when done:

```bash
bash scripts/stop_agentgym_servers.sh
```

## 6) Single-command CLIs (without wrappers)

Package data:

```bash
patchworld-package-data --source_root /path/to/splits --output_dir artifacts/patchworld/data_release --archive
```

Run benchmark (RQ1/RQ2 core):

```bash
patchworld-benchmark \
  --env maze \
  --train_glob artifacts/patchworld/data_release/maze/maze_traj_train.jsonl \
  --eval_glob artifacts/patchworld/data_release/maze/maze_traj_test.jsonl \
  --output_dir artifacts/patchworld/results/custom \
  --run_rollout
```

Run RQ3 on a specific model:

```bash
patchworld-rq3 \
  --env maze \
  --model_path artifacts/patchworld/results/rq1/generated_models/maze_patchworld_YYYYMMDD_HHMMSS.py \
  --num_tasks 200 \
  --max_steps 30
```
