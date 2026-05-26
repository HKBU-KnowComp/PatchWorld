from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from patchworld.inducer_llm_client import LLMClient
from patchworld.world_model_agent import BaseAgent, WorldModelAgent, _make_env_client, run_episode
from patchworld.worldmodel_validator import load_model_from_code


def _summarize(episodes: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not episodes:
        return {"n": 0, "success_rate": 0.0, "mean_reward": 0.0, "mean_steps": 0.0}
    n = len(episodes)
    success_rate = sum(1 for e in episodes if bool(e.get("success"))) / float(n)
    mean_reward = sum(float(e.get("total_reward", 0.0)) for e in episodes) / float(n)
    mean_steps = sum(int(e.get("steps", 0)) for e in episodes) / float(n)
    return {
        "n": n,
        "success_rate": round(success_rate, 4),
        "mean_reward": round(mean_reward, 4),
        "mean_steps": round(mean_steps, 2),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PatchWorld RQ3 live-agent evaluation.")
    parser.add_argument("--env", required=True, help="Environment name.")
    parser.add_argument("--model_path", required=True, help="Path to induced world model .py file.")
    parser.add_argument(
        "--llm_model",
        default="Qwen/Qwen3-Coder-480B-A35B-Instruct-Turbo",
        help="LLM model used by agent decision prompts at runtime.",
    )
    parser.add_argument("--num_tasks", type=int, default=200, help="Max number of tasks to evaluate.")
    parser.add_argument("--task_start", type=int, default=0, help="Task start index.")
    parser.add_argument("--max_steps", type=int, default=30, help="Max steps per episode.")
    parser.add_argument("--data_len", type=int, default=200, help="Task pool length for env client.")
    parser.add_argument("--env_server_base", default=None, help="Override environment server base URL.")
    parser.add_argument("--react_baseline", action="store_true", help="Also evaluate ReAct baseline.")
    parser.add_argument("--save_episodes", action="store_true", help="Store per-episode details in output JSON.")
    parser.add_argument(
        "--output_json",
        default=None,
        help="Where to save the RQ3 report JSON. Default: artifacts/patchworld/results/rq3_<env>_<timestamp>.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = Path(args.model_path)
    if not model_path.exists():
        raise SystemExit(f"model path not found: {model_path}")

    code = model_path.read_text(encoding="utf-8")
    world_model, load_error = load_model_from_code(code)
    if load_error or world_model is None:
        raise SystemExit(f"failed to load world model: {load_error}")

    llm = LLMClient(model=args.llm_model)
    task, client = _make_env_client(args.env, env_server_base=args.env_server_base, data_len=args.data_len)
    _ = task  # keeps task object alive

    n_available = len(client)
    end = min(args.task_start + args.num_tasks, n_available)
    task_indices = list(range(args.task_start, end))
    if not task_indices:
        raise SystemExit(
            f"no tasks to evaluate: available={n_available}, start={args.task_start}, num_tasks={args.num_tasks}"
        )

    wm_agent = WorldModelAgent(args.env, llm, world_model, max_steps=args.max_steps)
    wm_episodes = [run_episode(wm_agent, client, idx, max_steps=args.max_steps, verbose=False) for idx in task_indices]

    react_episodes: List[Dict[str, Any]] = []
    if args.react_baseline:
        react_agent = BaseAgent(args.env, llm, max_steps=args.max_steps)
        react_episodes = [
            run_episode(react_agent, client, idx, max_steps=args.max_steps, verbose=False) for idx in task_indices
        ]

    report: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "env": args.env,
        "model_path": str(model_path),
        "llm_model": args.llm_model,
        "task_start": args.task_start,
        "num_tasks_requested": args.num_tasks,
        "num_tasks_available": n_available,
        "task_indices": task_indices,
        "world_model_agent": {
            "summary": _summarize(wm_episodes),
            "episodes": wm_episodes if args.save_episodes else [],
        },
    }
    if args.react_baseline:
        report["react_baseline"] = {
            "summary": _summarize(react_episodes),
            "episodes": react_episodes if args.save_episodes else [],
        }

    if args.output_json:
        out_path = Path(args.output_json)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = Path(f"artifacts/patchworld/results/rq3_{args.env}_{ts}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"[patchworld-rq3] report: {out_path}")
    print(f"[patchworld-rq3] world_model_agent summary: {report['world_model_agent']['summary']}")
    if args.react_baseline:
        print(f"[patchworld-rq3] react_baseline summary: {report['react_baseline']['summary']}")


if __name__ == "__main__":
    main()
