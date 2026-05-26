import json
from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Iterable


@dataclass
class Transition:
    observation: str
    action: str
    next_observation: str
    reward: float
    done: bool


@dataclass
class Trajectory:
    transitions: List[Transition]
    env: str
    item_id: str
    task_idx: int
    rollout_index: int
    split: str
    success: bool
    total_reward: float

    @property
    def task_description(self) -> str:
        return self.transitions[0].observation if self.transitions else ""

    @property
    def num_steps(self) -> int:
        return len(self.transitions)


def _load_trajectories_from_file(jsonl_path: Path) -> List[Trajectory]:
    trajectories: List[Trajectory] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            meta = data.get("metadata", {})
            transitions_data = data.get("transitions", [])
            transitions = [
                Transition(
                    observation=t.get("observation", ""),
                    action=t.get("action", ""),
                    next_observation=t.get("next_observation", ""),
                    reward=float(t.get("reward", 0.0)),
                    done=bool(t.get("done", False)),
                )
                for t in transitions_data
            ]
            if not meta:
                # Fallback to minimal metadata when older formats are used.
                meta = {
                    "env": data.get("environment", ""),
                    "item_id": "",
                    "task_idx": -1,
                    "rollout_index": 0,
                    "split": "train",
                    "success": False,
                    "total_reward": sum(t.reward for t in transitions),
                }

            trajectories.append(
                Trajectory(
                    transitions=transitions,
                    env=str(meta.get("env", "")),
                    item_id=str(meta.get("item_id", "")),
                    task_idx=int(meta.get("task_idx", -1)),
                    rollout_index=int(meta.get("rollout_index", 0)),
                    split=str(meta.get("split", "train")),
                    success=bool(meta.get("success", False)),
                    total_reward=float(meta.get("total_reward", 0.0)),
                )
            )
    return trajectories


def load_trajectories(paths: Iterable[Path]) -> List[Trajectory]:
    """Load trajectories from one or more JSONL files."""
    all_trajs: List[Trajectory] = []
    for p in paths:
        if not p.exists():
            continue
        all_trajs.extend(_load_trajectories_from_file(p))
    return all_trajs


def group_by_problem(trajectories: List[Trajectory]) -> Dict[str, List[Trajectory]]:
    groups: Dict[str, List[Trajectory]] = defaultdict(list)
    for traj in trajectories:
        if traj.item_id:
            groups[traj.item_id].append(traj)
    for item_id in groups:
        groups[item_id].sort(key=lambda t: t.rollout_index)
    return dict(groups)


def group_by_task_type(trajectories: List[Trajectory]) -> Dict[str, List[Trajectory]]:
    groups: Dict[str, List[Trajectory]] = defaultdict(list)
    for traj in trajectories:
        desc = traj.task_description.split(".")[0].strip().lower()
        groups[desc].append(traj)
    return dict(groups)


def get_stats(trajectories: List[Trajectory]) -> Dict:
    problems = group_by_problem(trajectories)
    task_types = group_by_task_type(trajectories)
    n = len(trajectories)
    success_rate = (
        float(sum(1 for t in trajectories if t.success)) / float(n) if n > 0 else 0.0
    )
    return {
        "total_trajectories": n,
        "total_problems": len(problems),
        "task_types": len(task_types),
        "success_rate": success_rate,
        "task_type_samples": {k: len(v) for k, v in task_types.items()},
    }

