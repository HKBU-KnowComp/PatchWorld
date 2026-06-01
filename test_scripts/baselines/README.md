# PatchWorld Baseline Runners

This directory keeps comparative baseline orchestration next to PatchWorld
without vendoring the full baseline implementations. The launchers use
PatchWorld data and artifact defaults, then call the existing baseline code in
`/data/jbai/abduct-world` and, for PoE-World/WorldCoder, the synthesis stack in
`/data/jbai/poe-world`.

Only the paper-aligned RQ runners are exposed here:

- `rq1`: held-out one-step observation fidelity.
- `rq2`: autoregressive rollout fidelity.
- `rq3`: shared one-step lookahead planner.
- `train`: Qwen observation SFT training for the Word2World row.

The old per-environment AgentGym PoE-World and WorldCoder live evaluators are
intentionally not wired in because they use a different experimental setup.

## Quick Start

From `/data/jbai/patchworld`:

```bash
python test_scripts/baselines/run.py --baseline llm-direct --smoke
python test_scripts/baselines/run.py --baseline word2world --rq train --smoke
python test_scripts/baselines/run.py --baseline word2world --rq rq12 --smoke
python test_scripts/baselines/run.py --baseline word2world --rq rq3 --smoke

# These need the PoE-World/WorldCoder environment.
conda activate poeworld
python test_scripts/baselines/run.py --baseline poeworld --rq rq12 --smoke
python test_scripts/baselines/run.py --baseline poeworld --rq rq3 --smoke
python test_scripts/baselines/run.py --baseline worldcoder --rq rq12 --smoke
python test_scripts/baselines/run.py --baseline worldcoder --rq rq3 --smoke
```

Restrict a run with `--rq train`, `--rq rq1`, `--rq rq2`, `--rq rq12`, or
`--rq rq3`.
Baseline names also accept aliases:

- `llm_direct`, `llm-direct`
- `world2word`, `word2world`
- `poe-world`, `poeworld`
- `worldcoder-v2`, `worldcoder`

## Defaults

The launchers default to:

- `DATA_ROOT=/data/jbai/patchworld/artifacts/patchworld/data_release`
- `PATCHWORLD_BASELINE_ARTIFACTS_ROOT=/data/jbai/patchworld/artifacts/patchworld/baselines`
- `ABDUCTWORLD_REPO=/data/jbai/abduct-world`
- `POE_WORLD_REPO=/data/jbai/poe-world`
- `POE_WORLD_REPO_URL=https://github.com/marcos0318/poe-world.git`
- `ENVS=alfworld,babyai,maze,sciworld,textcraft,webshop,wordle`
- `MODEL=Qwen/Qwen3-Coder-480B-A35B-Instruct-Turbo`

Set `DEEPINFRA_API_KEY` or `PATCHWORLD_LLM_API_KEY` for LLM-backed baselines.
The runners also check `.deepinfra_api_key` in PatchWorld, abduct-world, and
the home directory.

## Revised PoE-World

PoE-World and WorldCoder should use the revised implementation with bug fixes:

```bash
git clone --recursive https://github.com/marcos0318/poe-world.git /data/jbai/poe-world
```

If you keep it somewhere else, set `POE_WORLD_REPO=/path/to/poe-world`. The
launchers warn when the local repo origin does not look like
`marcos0318/poe-world`.

## Baselines

### LLM-Direct

Runs RQ1/RQ2 through `test_scripts.abductworld.run_llm_direct_benchmark` and
RQ3 through the shared planner in `abductworld.evaluate_rq3`:

```bash
python test_scripts/baselines/run.py --baseline llm-direct --rq rq12
python test_scripts/baselines/run.py --baseline llm-direct --rq rq3
```

Useful overrides:

```bash
ENVS=maze MAX_TRAIN=30 MAX_EVAL_OBS=100 \
python test_scripts/baselines/run.py --baseline llm-direct --rq rq1
```

### Word2World

Runs the paper Word2World row through the abduct-world Qwen observation SFT
implementation in `test_scripts/qwen_observation/`, not the standalone
Word2World repo and not the older retrieval approximation.

Train one Qwen observation adapter per environment group:

```bash
python test_scripts/baselines/run.py --baseline word2world --rq train
```

Then run RQ1/RQ2 and RQ3 with the trained checkpoint root:

```bash
MODEL_DIR=artifacts/patchworld/baselines/results/qwen_observation_sft \
python test_scripts/baselines/run.py --baseline word2world --rq rq12

MODEL_DIR=artifacts/patchworld/baselines/results/qwen_observation_sft \
python test_scripts/baselines/run.py --baseline word2world --rq rq3
```

By default, training uses `run_train_qwen_observation_4cards.sh`, matching the
multi-card schedule in abduct-world. Set `WORD2WORLD_TRAIN_MODE=single` to use
`run_train_qwen_observation.sh` instead. Smoke mode sets
`DRY_RUN_BUILD_DATA=1` and small example caps, so it validates data construction
without launching a full model training job.

### PoE-World

Runs paper-aligned PoE-World RQ1/RQ2 through
`test_scripts.programmatic.run_programmatic_rq12` and RQ3 through
`abductworld.evaluate_rq3`. For RQ3, run RQ1/RQ2 first so
`world_models/poe_world_<env>_*.py` artifacts exist under
`PATCHWORLD_BASELINE_ARTIFACTS_ROOT/results/programmatic_rq12`.
AgentGym servers must also be running for RQ3.

```bash
conda activate poeworld
python test_scripts/baselines/run.py --baseline poeworld --rq rq12
python test_scripts/baselines/run.py --baseline poeworld --rq rq3
```

### WorldCoder

Runs paper-aligned WorldCoder RQ1/RQ2 through
`test_scripts.programmatic.run_programmatic_rq12` and RQ3 through
`abductworld.evaluate_rq3`. This is the poe-world-based WorldCoder stack used by
the paper, not the legacy `/data/jbai/WorldCoder` runner. Use the same revised
`POE_WORLD_REPO`, `poeworld` conda env, and RQ3 AgentGym server setup as
PoE-World.

```bash
conda activate poeworld
python test_scripts/baselines/run.py --baseline worldcoder --rq rq12
python test_scripts/baselines/run.py --baseline worldcoder --rq rq3
```

## Forwarding Extra Args

Extra arguments after `--` are forwarded to the selected shell launcher and then
to the underlying Python command when that launcher supports it:

```bash
python test_scripts/baselines/run.py --baseline llm-direct --rq rq1 -- --report_train_metrics
```
