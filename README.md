# PatchWorld

Standalone implementation of the PatchWorld code-based world-model induction
pipeline used in the paper experiments.

It includes:
- LLM-driven symbolic world-model induction with TracePatch-style repair.
- Replay-based validation and model selection.
- One-step, rollout, and live-agent (RQ3) evaluation utilities.
- Optional train-only residual memory wrapper.

## Install

Default: create a venv and install into it.

```bash
cd /path/to/patchworld
python -m venv .venv
source .venv/bin/activate
bash scripts/install.sh
```

Or create a named conda env in one step:

```bash
cd /path/to/patchworld
PATCHWORLD_CONDA_ENV=patchworld bash scripts/install.sh
conda activate patchworld
```

`scripts/install.sh` installs PatchWorld plus the **AgentGym client from GitHub**
(PyPI releases are stale; do not `pip install agentenv`):

```bash
pip install -e .
pip install "git+https://github.com/marcos0318/AgentGym.git#subdirectory=agentenv" --no-deps
python -m nltk.downloader punkt
```

The `--no-deps` flag keeps AgentGym from overwriting PatchWorld's runtime
packages (`torch`, `transformers`, `requests`, etc.).

Override the GitHub source if needed:

```bash
AGENTGYM_GIT_URL=git+https://github.com/marcos0318/AgentGym.git#subdirectory=agentenv \
bash scripts/install.sh
```

## Required Data Format

PatchWorld expects JSONL trajectory files with:
- `metadata`: `env`, `item_id`, `task_idx`, `rollout_index`, `split`, `success`, `total_reward`
- `transitions`: list of `{observation, action, next_observation, reward, done}`

## Quick Start

Download the trajectory splits first (see [Download Data](#download-data-hugging-face)).

Induce + evaluate one environment:

```bash
patchworld-benchmark \
  --env maze \
  --train_glob "artifacts/patchworld/data_release/maze/maze_traj_train.jsonl" \
  --eval_glob "artifacts/patchworld/data_release/maze/maze_traj_test.jsonl" \
  --output_dir artifacts/patchworld/results
```

Induce model only:

```bash
patchworld-induce \
  --env maze \
  --train_glob "artifacts/patchworld/data_release/maze/maze_traj_train.jsonl" \
  --output_model artifacts/patchworld/generated/maze_model.py
```

## Download Data (Hugging Face)

Trajectory splits are **not** stored in this GitHub repo. Download them from Hugging Face:

- Dataset: [HKBU-KnowComp/patchworld-trajectories](https://huggingface.co/datasets/HKBU-KnowComp/patchworld-trajectories)

Install the Hugging Face CLI if needed:

```bash
pip install -U huggingface_hub
```

Download into the default location used by the experiment scripts:

```bash
bash scripts/download_data.sh
```

This writes files to `artifacts/patchworld/data_release/`.

Manual download:

```bash
hf download HKBU-KnowComp/patchworld-trajectories \
  --repo-type dataset \
  --local-dir artifacts/patchworld/data_release \
  --exclude "*.tar.gz"
```

Expected layout after download:

- `artifacts/patchworld/data_release/<env>/<env>_traj_{train,val,test}.jsonl`
- `artifacts/patchworld/data_release/manifest.json`

The helper scripts default to `DATA_ROOT=artifacts/patchworld/data_release`.

If you maintain the dataset, upload from a packaged local copy with:

```bash
bash scripts/upload_data.sh
```

## Experiment Commands

Use the helper scripts for paper-style runs:

```bash
# RQ1: one-step fidelity
bash scripts/run_rq1.sh

# RQ2: rollout robustness
bash scripts/run_rq2.sh

# RQ3: live planning (requires AgentGym install + servers)
bash scripts/run_rq3.sh
```

For full details, see `EXPERIMENTS.md`.

## Comparative Baselines

Baseline launchers for LLM-Direct, Word2World, PoE-World, and WorldCoder live in
`test_scripts/baselines/`. They expose the paper-aligned comparisons, default to
PatchWorld's downloaded trajectory splits, and write to
`artifacts/patchworld/baselines/`. Word2World also exposes the Qwen observation
SFT training path from `/data/jbai/abduct-world/test_scripts/qwen_observation`.

```bash
python test_scripts/baselines/run.py --baseline llm-direct --smoke
python test_scripts/baselines/run.py --baseline word2world --rq train --smoke
python test_scripts/baselines/run.py --baseline word2world --rq rq12 --smoke
```

PoE-World and WorldCoder use the revised bug-fixed implementation from
`https://github.com/marcos0318/poe-world` via `POE_WORLD_REPO` and should be run
from the separate `poeworld` conda environment:

```bash
conda activate poeworld
python test_scripts/baselines/run.py --baseline poeworld --rq rq12 --smoke
python test_scripts/baselines/run.py --baseline poeworld --rq rq3 --smoke
python test_scripts/baselines/run.py --baseline worldcoder --rq rq12 --smoke
python test_scripts/baselines/run.py --baseline worldcoder --rq rq3 --smoke
```

See `test_scripts/baselines/README.md` for all overrides.

## Environment Variables

LLM API settings (OpenAI-compatible):
- `PATCHWORLD_LLM_API_KEY` (or `DEEPINFRA_API_KEY`)
- `PATCHWORLD_LLM_BASE_URL` (optional)
- `PATCHWORLD_LLM_CACHE=1` to enable SQLite cache

## RQ3 Prerequisite (Live Planning)

PatchWorld itself (including the AgentGym HTTP client) is installed by
`bash scripts/install.sh` above. RQ3 additionally needs **separate conda envs
per AgentGym server** — do not run multiple servers from one shared conda env.

### 1) Install AgentGym server envs (one conda env each)

Clone AgentGym once if you have not already:

```bash
git clone --recursive https://github.com/marcos0318/AgentGym ../AgentGym
```

Then bootstrap the core paper servers:

```bash
bash scripts/install_agentgym_envs.sh
```

| Conda env | Server package | Notes |
|---|---|---|
| `agentenv-alfworld` | `agentenv-alfworld` | runs `setup.sh` |
| `agentenv-sciworld` | `agentenv-sciworld` | Python 3.8 |
| `agentenv-babyai` | `agentenv-babyai` | |
| `agentenv-lmrlgym` | `agentenv-lmrlgym` | maze/wordle; uses `environment.yml` |
| `agentenv-textcraft` | `agentenv-textcraft` | launch from package dir |
| `agentenv-webshop` | `agentenv-webshop` | uses `environment.yml` + `setup.sh` |

Install a subset only:

```bash
ONLY_ENVS=lmrlgym bash scripts/install_agentgym_envs.sh
```

Custom conda env names (use the same names when starting servers):

```bash
CONDA_ENV_LMRLGYM=pw-test-lmrlgym \
ONLY_ENVS=lmrlgym \
bash scripts/install_agentgym_envs.sh

ONLY_SERVERS=lmrlgym \
CONDA_ENV_LMRLGYM=pw-test-lmrlgym \
bash scripts/start_agentgym_servers.sh
```

Available overrides: `CONDA_ENV_ALFWORLD`, `CONDA_ENV_SCIWORLD`,
`CONDA_ENV_BABYAI`, `CONDA_ENV_LMRLGYM`, `CONDA_ENV_TEXTCRAFT`,
`CONDA_ENV_WEBSHOP`.

Per-server launch commands are documented in each `agentenv-*` package under
the [AgentGym](https://github.com/marcos0318/AgentGym) repository.

### 2) Start servers

```bash
bash scripts/start_agentgym_servers.sh
```

Stop all servers:

```bash
bash scripts/stop_agentgym_servers.sh
```

Default port mapping:
- `36001`: alfworld
- `36002`: sciworld
- `36003`: babyai
- `36004`: lmrlgym (maze/wordle)
- `36005`: textcraft
- `36006`: webshop

### 3) Run RQ3

Uses the latest induced model per env from RQ1 output by default:

```bash
bash scripts/run_rq3.sh
```

Quick smoke test (3 tasks, maze only):

```bash
ENVS=maze RQ1_DIR=artifacts/patchworld/results/rq1_smoke \
OUT_DIR=artifacts/patchworld/results/rq3_smoke \
NUM_TASKS=3 MAX_STEPS=10 \
bash scripts/run_rq3.sh
```

## Repository Layout

- `patchworld/`: core package (induction, validator, evaluator, residual memory)
- `patchworld/cli/`: standalone experiment CLIs
- `scripts/`: AgentGym install and server-management helpers for RQ3
- `EXPERIMENTS.md`: end-to-end runbook for RQ1/RQ2/RQ3
- `examples/generated_world_models/`: appendix example induced models

## License

MIT — see [LICENSE](LICENSE).
