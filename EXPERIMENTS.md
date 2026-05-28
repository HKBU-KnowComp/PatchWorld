# PatchWorld Experiments Runbook

This document describes how to package data and run PatchWorld experiments
end-to-end in the standalone repo.

## 1) Setup

```bash
cd /path/to/patchworld
PATCHWORLD_CONDA_ENV=patchworld bash scripts/install.sh
conda activate patchworld
```

Venv alternative:

```bash
python -m venv .venv
source .venv/bin/activate
bash scripts/install.sh
```

This installs PatchWorld and the AgentGym client from GitHub (required; PyPI is stale):

```bash
pip install -e .
pip install "git+https://github.com/marcos0318/AgentGym.git#subdirectory=agentenv" --no-deps
python -m nltk.downloader punkt
```

Set API key for induction/evaluation:

```bash
export PATCHWORLD_LLM_API_KEY=...
# or use DEEPINFRA_API_KEY
```

## 2) Get the Data

### Download from Hugging Face (recommended)

Trajectory splits are hosted on Hugging Face, not GitHub:

- [HKBU-KnowComp/patchworld-trajectories](https://huggingface.co/datasets/HKBU-KnowComp/patchworld-trajectories)

```bash
pip install -U huggingface_hub
bash scripts/download_data.sh
```

This writes to `artifacts/patchworld/data_release/`.

### Package locally (maintainers only)

If you already have split JSONLs locally, package them for upload (set `SOURCE_ROOT`):

```bash
SOURCE_ROOT=/path/to/split_jsonls bash scripts/package_data.sh
```

Upload to Hugging Face (requires a write token):

```bash
export HF_TOKEN=...   # token with write access
bash scripts/upload_data.sh
```

Overrides for packaging:

```bash
SOURCE_ROOT=/path/to/resplit_train_val_test_seed42 \
OUTPUT_DIR=artifacts/patchworld/data_release \
ENVS=alfworld,babyai,maze,sciworld,textcraft,webshop,wordle \
bash scripts/package_data.sh
```

Packaging output:
- `artifacts/patchworld/data_release/<env>/<env>_traj_{train,val,test}.jsonl`
- `artifacts/patchworld/data_release/manifest.json`
- `artifacts/patchworld/data_release_<timestamp>.tar.gz` (local archive only; not uploaded)

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

### 5.1 Install AgentGym server envs

PatchWorld + the AgentGym client are installed in section 1). RQ3 also needs
**one conda env per AgentGym server**:

```bash
git clone --recursive https://github.com/marcos0318/AgentGym ../AgentGym
bash scripts/install_agentgym_envs.sh
```

Maze/wordle only:

```bash
ONLY_ENVS=lmrlgym bash scripts/install_agentgym_envs.sh
```

Custom server env names:

```bash
CONDA_ENV_LMRLGYM=pw-test-lmrlgym ONLY_ENVS=lmrlgym bash scripts/install_agentgym_envs.sh
ONLY_SERVERS=lmrlgym CONDA_ENV_LMRLGYM=pw-test-lmrlgym bash scripts/start_agentgym_servers.sh
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
