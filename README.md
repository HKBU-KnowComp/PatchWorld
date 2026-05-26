# PatchWorld (Anonymous Standalone Release)

This repository is a standalone, anonymized implementation of the PatchWorld
code-based world-model induction pipeline used in the paper experiments.

It includes:
- LLM-driven symbolic world-model induction with TracePatch-style repair.
- Replay-based validation and model selection.
- One-step and rollout evaluation utilities.
- Optional train-only residual memory wrapper.

Current CLI coverage focuses on offline induction/evaluation (RQ1/RQ2-style).
If you run a live-agent RQ3 setup, use the AgentGym source installation
requirements below.

## Install

```bash
cd /path/to/patchworld
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Install optional NLTK resources used by BLEU metrics:

```bash
python -m nltk.downloader punkt
```

## Required Data Format

PatchWorld expects JSONL trajectory files with:
- `metadata`: `env`, `item_id`, `task_idx`, `rollout_index`, `split`, `success`, `total_reward`
- `transitions`: list of `{observation, action, next_observation, reward, done}`

## Quick Start

Induce + evaluate one environment:

```bash
patchworld-benchmark \
  --env maze \
  --train_glob "artifacts/resplit_train_val_test_seed42/maze/maze_traj_train.jsonl" \
  --eval_glob "artifacts/resplit_train_val_test_seed42/maze/maze_traj_test.jsonl" \
  --output_dir artifacts/patchworld/results
```

Induce model only:

```bash
patchworld-induce \
  --env maze \
  --train_glob "artifacts/resplit_train_val_test_seed42/maze/maze_traj_train.jsonl" \
  --output_model artifacts/patchworld/generated/maze_model.py
```

## Data Packaging

Create an anonymized data bundle from existing split JSONLs:

```bash
bash scripts/package_data.sh
```

Default source:
- `../abduct-world/artifacts/resplit_train_val_test_seed42`

Default output:
- `artifacts/patchworld/data_release/`

The packer writes:
- per-env split files (`*_traj_train.jsonl`, `*_traj_val.jsonl`, `*_traj_test.jsonl`)
- `manifest.json` with checksums and counts
- a timestamped `.tar.gz` archive

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

## Environment Variables

LLM API settings (OpenAI-compatible):
- `PATCHWORLD_LLM_API_KEY` (or `DEEPINFRA_API_KEY`)
- `PATCHWORLD_LLM_BASE_URL` (optional)
- `PATCHWORLD_LLM_CACHE=1` to enable SQLite cache

## RQ3 Prerequisite (Live Planning)

For RQ3-style live planning evaluation against AgentGym environments, use
AgentGym source installs (not only PyPI). In our reference setup,
install in this order:

1) install core `agentenv` from source first  
2) install required `agentenv-*` environment packages  
3) start all servers on the standard ports

### Recommended one-command install

```bash
bash scripts/install_agentgym_envs.sh
```

This script installs from `AGENTGYM_DIR` (default `../AgentGym`) and
follows the package order:

- `agentenv` (core, first)
- `agentenv-alfworld`
- `agentenv-sciworld`
- `agentenv-babyai`
- `agentenv-lmrlgym` (Maze/Wordle)
- `agentenv-textcraft`
- `agentenv-webshop`
- `agentenv-webarena`
- `agentenv-tool` (weather/todo/movie/sheet/academia)
- `agentenv-searchqa`
- `agentenv-sqlgym`

By default it uses environment-specific conda env names (`agentenv-alfworld`,
`agentenv-sciworld`, `agentenv-babyai`, `agentenv-lmrlgym`,
`agentenv-textcraft`, `agentenv-webshop`, `agentenv-webarena`,
`agentenv-tool`, `agentenv-searchqa`, `agentenv-sqlgym`) because these
packages have different Python constraints in AgentGym.

### Manual source install (if you prefer)

```bash
git clone --recursive https://github.com/marcos0318/AgentGym
cd AgentGym
pip install -e ./agentenv
pip install -e ./agentenv-alfworld
pip install -e ./agentenv-sciworld
pip install -e ./agentenv-babyai
pip install -e ./agentenv-lmrlgym
pip install -e ./agentenv-textcraft
pip install -e ./agentenv-webshop
```

If you want to avoid dependency drift in existing envs, use `--no-deps` with
editable installs (the helper script defaults to this behavior).

### Start servers (all at once)

Paper/core environments:

```bash
bash scripts/start_agentgym_servers.sh
```

Full AgentGym ports (36001-36014):

```bash
PROFILE=full bash scripts/start_agentgym_servers.sh
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
- `36007`: webarena
- `36008`: weather
- `36009`: todo
- `36010`: movie
- `36011`: sheet
- `36012`: academia
- `36013`: searchqa
- `36014`: sqlgym

## Repository Layout

- `patchworld/`: core package (induction, validator, evaluator, residual memory)
- `patchworld/cli/`: standalone experiment CLIs
- `scripts/`: AgentGym install and server-management helpers for RQ3
- `EXPERIMENTS.md`: end-to-end runbook for RQ1/RQ2/RQ3
- `examples/generated_world_models/`: appendix example induced models

## Notes for Anonymous Review

- The code avoids project-specific naming and paths from prior internal repos.
- Outputs default to `artifacts/patchworld/...`.
- No baseline-specific external repos are required for core induction/evaluation.
