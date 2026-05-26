from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from glob import glob
from pathlib import Path
from typing import Dict, List

from patchworld.inducer_llm_client import LLMClient
from patchworld.residual_memory import EpisodicResidualWorldModel
from patchworld.worldmodel_data import Trajectory, get_stats, load_trajectories
from patchworld.worldmodel_evaluator import (
    compute_autoregressive_rollout_token_f1,
    compute_experience_fit,
    compute_observation_fit,
)
from patchworld.worldmodel_inducer import Inducer
from patchworld.worldmodel_validator import load_model_from_code


def _load_paths(pattern: str) -> List[Path]:
    paths = [Path(p) for p in glob(pattern)]
    return [p for p in sorted(paths) if p.is_file() and p.suffix == ".jsonl"]


def _load_trajs(pattern: str) -> List[Trajectory]:
    paths = _load_paths(pattern)
    if not paths:
        raise SystemExit(f"No trajectory files matched: {pattern}")
    trajs = load_trajectories(paths)
    if not trajs:
        raise SystemExit(f"No trajectories loaded from: {pattern}")
    return trajs


def _subsample(trajs: List[Trajectory], max_count: int, seed: int) -> List[Trajectory]:
    if max_count <= 0 or len(trajs) <= max_count:
        return trajs
    rnd = random.Random(seed)
    return rnd.sample(trajs, max_count)


def _evaluate_model(model: object, train: List[Trajectory], eval_: List[Trajectory], env: str) -> Dict[str, Dict]:
    return {
        "train": {
            "experience_fit": compute_experience_fit(model, train),
            "observation_fit": compute_observation_fit(model, train, env_name=env),
        },
        "eval": {
            "experience_fit": compute_experience_fit(model, eval_),
            "observation_fit": compute_observation_fit(model, eval_, env_name=env),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone PatchWorld benchmark runner.")
    parser.add_argument("--env", required=True, help="Environment name (e.g. maze, textcraft, webshop).")
    parser.add_argument("--train_glob", required=True, help="Glob for train JSONL trajectories.")
    parser.add_argument("--eval_glob", required=True, help="Glob for eval JSONL trajectories.")
    parser.add_argument(
        "--output_dir",
        default="artifacts/patchworld/results",
        help="Output directory for generated models and benchmark reports.",
    )
    parser.add_argument("--max_train", type=int, default=0, help="Max train trajectories; <=0 uses all.")
    parser.add_argument("--max_eval", type=int, default=0, help="Max eval trajectories; <=0 uses all.")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-Coder-480B-A35B-Instruct-Turbo",
        help="OpenAI-compatible model identifier for induction.",
    )
    parser.add_argument("--induction_strategy", default="tracepatch", choices=["tracepatch", "naive_induction"])
    parser.add_argument("--max_refine_rounds", type=int, default=15)
    parser.add_argument("--contrastive_max_per_pattern", type=int, default=5)
    parser.add_argument("--contrastive_max_transitions", type=int, default=60)
    parser.add_argument("--patch_candidates_per_round", type=int, default=4)
    parser.add_argument("--patch_beam_size", type=int, default=2)
    parser.add_argument("--patch_no_improve_patience", type=int, default=2)

    parser.add_argument("--run_rollout", action="store_true", help="Compute autoregressive rollout metrics (RQ2-style).")
    parser.add_argument("--rollout_horizon", type=int, default=15)
    parser.add_argument("--rollout_steps", default="1,2,3,5,8,10,15")

    parser.add_argument(
        "--use_residual_memory",
        action="store_true",
        help="Evaluate train-only residual memory wrapper in addition to base symbolic model.",
    )
    parser.add_argument("--residual_memory_max_entries", type=int, default=0)
    parser.add_argument("--residual_memory_min_count", type=int, default=1)
    parser.add_argument("--residual_memory_min_confidence", type=float, default=1.0)

    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    train_all = _load_trajs(args.train_glob)
    eval_all = _load_trajs(args.eval_glob)
    train_trajs = _subsample(train_all, args.max_train, args.seed)
    eval_trajs = _subsample(eval_all, args.max_eval, args.seed + 1)

    out_dir = Path(args.output_dir)
    model_dir = out_dir / "generated_models"
    model_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = model_dir / f"{args.env}_patchworld_{ts}.py"

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

    if args.verbose:
        print(
            f"[patchworld-benchmark] inducing model for env={args.env} "
            f"train={len(train_trajs)} eval={len(eval_trajs)}"
        )
    code = inducer.induce(args.env, train_trajs, validation_trajectories=eval_trajs)
    model_path.write_text(code, encoding="utf-8")

    model, load_error = load_model_from_code(code)
    if load_error or model is None:
        raise SystemExit(f"Generated model failed to load: {load_error}")

    report: Dict[str, object] = {
        "timestamp": datetime.now().isoformat(),
        "env": args.env,
        "model": args.model,
        "train_glob": args.train_glob,
        "eval_glob": args.eval_glob,
        "train_stats_pool": get_stats(train_all),
        "eval_stats_pool": get_stats(eval_all),
        "train_size_used": len(train_trajs),
        "eval_size_used": len(eval_trajs),
        "induction": {
            "strategy": args.induction_strategy,
            "max_refine_rounds": args.max_refine_rounds,
            "stats": dict(inducer.last_induce_stats or {}),
            "llm_usage": llm.get_usage(),
        },
        "symbolic_model_path": str(model_path),
        "simple_state": _evaluate_model(model, train_trajs, eval_trajs, args.env),
    }

    if args.use_residual_memory:
        residual = EpisodicResidualWorldModel(
            model,
            train_trajs,
            env_name=args.env,
            max_entries=args.residual_memory_max_entries,
            min_count=args.residual_memory_min_count,
            min_confidence=args.residual_memory_min_confidence,
        )
        report["simple_state_residual"] = _evaluate_model(residual, train_trajs, eval_trajs, args.env)
        report["residual_memory"] = {"stats": residual.residual_stats}

    if args.run_rollout:
        rollout_steps = [int(s.strip()) for s in args.rollout_steps.split(",") if s.strip()]
        report["rq2_rollout"] = {
            "simple_state": compute_autoregressive_rollout_token_f1(
                model,
                eval_trajs,
                max_horizon=args.rollout_horizon,
                report_steps=rollout_steps,
            )
        }
        if args.use_residual_memory and "simple_state_residual" in report:
            residual_model = EpisodicResidualWorldModel(
                model,
                train_trajs,
                env_name=args.env,
                max_entries=args.residual_memory_max_entries,
                min_count=args.residual_memory_min_count,
                min_confidence=args.residual_memory_min_confidence,
            )
            report["rq2_rollout"]["simple_state_residual"] = compute_autoregressive_rollout_token_f1(
                residual_model,
                eval_trajs,
                max_horizon=args.rollout_horizon,
                report_steps=rollout_steps,
            )

    report_path = out_dir / f"benchmark_{args.env}_{ts}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"[patchworld-benchmark] model:  {model_path}")
    print(f"[patchworld-benchmark] report: {report_path}")


if __name__ == "__main__":
    main()
