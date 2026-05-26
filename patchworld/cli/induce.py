from __future__ import annotations

import argparse
import json
from datetime import datetime
from glob import glob
from pathlib import Path
from typing import List

from patchworld.inducer_llm_client import LLMClient
from patchworld.worldmodel_data import Trajectory, get_stats, load_trajectories
from patchworld.worldmodel_inducer import Inducer


def _load_paths(pattern: str) -> List[Path]:
    paths = [Path(p) for p in glob(pattern)]
    return [p for p in sorted(paths) if p.is_file() and p.suffix == ".jsonl"]


def _load_trajectories_from_glob(pattern: str) -> List[Trajectory]:
    paths = _load_paths(pattern)
    if not paths:
        raise SystemExit(f"No trajectory files matched: {pattern}")
    trajs = load_trajectories(paths)
    if not trajs:
        raise SystemExit(f"No trajectories loaded from: {pattern}")
    return trajs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Induce a PatchWorld model from train trajectories.")
    parser.add_argument("--env", required=True, help="Environment name, e.g. maze, webshop, sciworld.")
    parser.add_argument("--train_glob", required=True, help="Glob pattern for train JSONL trajectories.")
    parser.add_argument("--output_model", required=True, help="Output path for generated Python model file.")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-Coder-480B-A35B-Instruct-Turbo",
        help="OpenAI-compatible model identifier used for induction.",
    )
    parser.add_argument("--max_refine_rounds", type=int, default=15)
    parser.add_argument("--induction_strategy", default="tracepatch", choices=["tracepatch", "naive_induction"])
    parser.add_argument("--contrastive_max_per_pattern", type=int, default=5)
    parser.add_argument("--contrastive_max_transitions", type=int, default=60)
    parser.add_argument("--patch_candidates_per_round", type=int, default=4)
    parser.add_argument("--patch_beam_size", type=int, default=2)
    parser.add_argument("--patch_no_improve_patience", type=int, default=2)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--stats_json",
        default=None,
        help="Optional JSON path to write induction metadata and model usage stats.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_trajs = _load_trajectories_from_glob(args.train_glob)
    print(f"[patchworld-induce] loaded {len(train_trajs)} train trajectories")
    print(f"[patchworld-induce] train stats: {json.dumps(get_stats(train_trajs), indent=2)}")

    llm = LLMClient(model=args.model)
    inducer = Inducer(
        llm=llm,
        max_refine_rounds=args.max_refine_rounds,
        contrastive_max_per_pattern=args.contrastive_max_per_pattern,
        contrastive_max_transitions=args.contrastive_max_transitions,
        induction_strategy=args.induction_strategy,
        patch_candidates_per_round=args.patch_candidates_per_round,
        patch_beam_size=args.patch_beam_size,
        patch_no_improve_patience=args.patch_no_improve_patience,
        verbose=args.verbose,
        show_progress=args.verbose,
        induction_track="simple_state",
    )

    code = inducer.induce(args.env, train_trajs)
    out_path = Path(args.output_model)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(code, encoding="utf-8")
    print(f"[patchworld-induce] wrote model to {out_path}")

    if args.stats_json:
        payload = {
            "timestamp": datetime.now().isoformat(),
            "env": args.env,
            "model": args.model,
            "train_glob": args.train_glob,
            "train_size": len(train_trajs),
            "output_model": str(out_path),
            "llm_usage": llm.get_usage(),
            "induce_stats": dict(inducer.last_induce_stats or {}),
        }
        stats_path = Path(args.stats_json)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[patchworld-induce] wrote stats to {stats_path}")


if __name__ == "__main__":
    main()
