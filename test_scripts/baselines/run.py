#!/usr/bin/env python3
"""Unified launcher for PatchWorld comparative baselines."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


BASELINE_SCRIPTS = {
    "llm-direct": "llm_direct.sh",
    "word2world": "word2world.sh",
    "poeworld": "poeworld.sh",
    "worldcoder": "worldcoder.sh",
}

ALIASES = {
    "llm_direct": "llm-direct",
    "llm-direct": "llm-direct",
    "world2word": "word2world",
    "word2world": "word2world",
    "poe-world": "poeworld",
    "poeworld": "poeworld",
    "worldcoder-v2": "worldcoder",
    "worldcoder": "worldcoder",
}


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "Run comparative baseline launchers from PatchWorld. Extra args after "
            "'--' are forwarded to the selected baseline script."
        )
    )
    parser.add_argument(
        "--baseline",
        required=True,
        choices=sorted(ALIASES.keys()),
        help="Baseline family to run.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use small default limits for a fast sanity check.",
    )
    parser.add_argument(
        "--rq",
        choices=["train", "rq1", "rq2", "rq12", "rq3", "all"],
        default=None,
        help="Restrict the selected baseline to one experiment family.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the command and resolved defaults without executing.",
    )
    return parser.parse_known_args()


def main() -> int:
    args, extra_args = parse_args()
    scripts_root = Path(__file__).resolve().parent
    repo_root = scripts_root.parent.parent
    baseline = ALIASES[args.baseline]
    script_path = scripts_root / BASELINE_SCRIPTS[baseline]

    env = os.environ.copy()
    env.setdefault("PATCHWORLD_BASELINE_ARTIFACTS_ROOT", str(repo_root / "artifacts/patchworld/baselines"))
    env.setdefault("DATA_ROOT", str(repo_root / "artifacts/patchworld/data_release"))

    if args.smoke:
        env["SMOKE_TEST"] = "1"
        env.setdefault("ENVS", "maze")
        env.setdefault("MAX_PARALLEL", "1")
        env.setdefault("MAX_SAMPLES", "3")
        env.setdefault("NUM_TRAIN", "3")
        env.setdefault("NUM_EVAL", "3")
        env.setdefault("AGENT_NUM_TASKS", "3")
        env.setdefault("AGENT_TASK_CAP", "3")

    if args.rq:
        env["BASELINE_RQ"] = args.rq

    cmd = ["bash", str(script_path), *extra_args]
    print(f"[patchworld-baselines] baseline={baseline}")
    print(f"[patchworld-baselines] rq={env.get('BASELINE_RQ', 'all')}")
    print(f"[patchworld-baselines] command={' '.join(cmd)}")
    print(f"[patchworld-baselines] DATA_ROOT={env['DATA_ROOT']}")
    print(f"[patchworld-baselines] PATCHWORLD_BASELINE_ARTIFACTS_ROOT={env['PATCHWORLD_BASELINE_ARTIFACTS_ROOT']}")
    if args.dry_run:
        return 0

    completed = subprocess.run(cmd, cwd=repo_root, env=env, check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
