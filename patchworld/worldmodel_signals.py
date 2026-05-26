from __future__ import annotations

from typing import Any, Dict, List


# Shared failure-indicator strings used to detect when an action actually failed.
# These are checked as substrings (case-insensitive) of next_observation.
FAILURE_INDICATORS: List[str] = [
    "no known action matches",
    "the door is not open",
    "nothing happens",
    "i can't",
    "you can't",
    "invalid action",
    "i don't understand",
    "that's not something you can",
    "you don't have",
    "there is no",
    "already",
]


def action_actually_failed(next_obs: str) -> bool:
    lo = (next_obs or "").lower()
    return any(ind in lo for ind in FAILURE_INDICATORS)


def belief_snapshot(belief: Any) -> Dict[str, Any]:
    """Return a lightweight, serializable snapshot of arbitrary belief objects."""
    if isinstance(belief, dict):
        return dict(belief)

    metadata = getattr(belief, "metadata", None)
    nodes = getattr(belief, "nodes", None)
    edges = getattr(belief, "edges", None)
    snapshot: Dict[str, Any] = {"belief_type": type(belief).__name__}
    if isinstance(metadata, dict):
        snapshot["metadata"] = dict(metadata)
    if isinstance(nodes, dict):
        snapshot["node_count"] = len(nodes)
    if edges is not None:
        try:
            snapshot["edge_count"] = len(edges)
        except Exception:
            pass
    return snapshot
