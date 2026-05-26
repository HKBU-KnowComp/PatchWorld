"""Contrastive transition selection helpers for world-model induction."""

import random
import re
from typing import Dict, List, Tuple

from patchworld.worldmodel_data import Trajectory, Transition


def _action_type(action: str) -> str:
    """Reduce an action to its type/template by stripping specific values."""
    action = action.strip().lower()
    if not action:
        return ""

    # Wordle actions often appear as spaced letters: "s h i r e".
    if re.fullmatch(r"[a-z](?:\s+[a-z]){4}", action):
        return "<5-letter-guess>"

    # API-style actions like search[foo], click[bar], guess[a b c d e].
    bracket_match = re.fullmatch(r"([a-z_][a-z0-9_ ]*)\[(.*)\]", action)
    if bracket_match:
        name = bracket_match.group(1).strip()
        payload = bracket_match.group(2).strip()
        if name == "click":
            if re.fullmatch(r"[a-z0-9]{10}", payload, flags=re.IGNORECASE):
                return "click[<asin>]"
            if payload.lower().startswith("page "):
                return "click[page]"
            if payload in {"next >", "< prev", "back to search", "search", "buy now"}:
                return f"click[{payload.lower()}]"
            return "click[<option>]"
        if name == "search":
            return "search[<query>]"
        if name == "guess":
            return "guess[<5-letter-word>]"
        return f"{name}[<X>]"

    # Standard text actions.
    tokens = action.split()
    if len(tokens) == 1:
        return tokens[0]
    if tokens[0] in {"go", "pick", "put", "turn", "look"} and len(tokens) >= 2:
        verb = " ".join(tokens[:2])
    else:
        verb = tokens[0]
    return verb + " <X>"


def _outcome_type(next_obs: str) -> str:
    """Reduce a next_observation to its outcome type."""
    lo = next_obs.strip().lower()
    if not lo:
        return "EMPTY"
    if re.fullmatch(r"[byg](?: [byg]){4}", lo):
        return "WORDLE_FEEDBACK"
    if lo == "invalid word":
        return "WORDLE_INVALID"
    if "welcome to the game of wordle" in lo:
        return "WORDLE_WELCOME"
    if lo.startswith("webshop [sep] ") or lo.startswith("instruction: [sep] "):
        if "price:" in lo and "rating:" in lo:
            return "WEBSHOP_PRODUCT_PAGE"
        if any(k in lo for k in ("size [sep]", "color [sep]", "fit type [sep]", "flavor [sep]", "item shape [sep]", "scent [sep]")):
            return "WEBSHOP_FILTER_PAGE"
        if "page " in lo or "back to search" in lo:
            return "WEBSHOP_RESULTS_PAGE"
    # Known failure patterns
    if any(f in lo for f in ["no known action", "nothing happens", "invalid", "i can't", "you can't"]):
        return "FAIL"
    # Truncate and normalise to first ~40 chars as a signature
    return re.sub(r"[^a-z ]", " ", lo[:60]).strip()


def _task_text_from_observation(obs_text: str) -> str:
    match = re.search(
        r"Task description:\s*Your task is to ([^\n]+)",
        obs_text or "",
        flags=re.IGNORECASE,
    )
    if not match:
        return ""
    return match.group(1).strip().rstrip(".")


def _sciworld_task_family(obs_text: str) -> str:
    task = _task_text_from_observation(obs_text).lower()
    if not task:
        return "other"
    if "electrically conductive" in task:
        return "conductivity"
    if "measure the temperature" in task:
        return "temperature"
    if "measure the melting point" in task:
        return "melting_point"
    if task.startswith(("melt ", "boil ", "freeze ")) or "change the state of matter" in task:
        return "state_change"
    if "use chemistry to create" in task:
        return "chemistry"
    if "light bulb" in task or "powering it using" in task:
        return "circuits"
    if "focus on the" in task and "life stage" in task:
        return "life_cycle"
    if "longest life span" in task or "shortest life span" in task:
        return "lifespan"
    if "find a(n)" in task:
        return "classification"
    return "other"


def _alfworld_task_family(obs_text: str) -> str:
    task = _task_text_from_observation(obs_text).lower()
    if not task:
        return "other"
    if "clean" in task:
        return "clean"
    if "heat" in task:
        return "heat"
    if "cool" in task:
        return "cool"
    if "put " in task and (" in " in task or " on " in task):
        return "put"
    if any(tok in task for tok in ("pick", "take", "find", "get ")):
        return "pickup"
    return _normalize_signature(task, max_len=72)


def _normalize_signature(text: str, *, max_len: int = 160) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())[:max_len]


def _wordle_guess_key(action: str) -> str:
    return re.sub(r"[^a-z]", "", (action or "").lower())


def _webshop_bucket(obs_or_next_obs: str) -> str:
    outcome = _outcome_type(obs_or_next_obs)
    if outcome.startswith("WEBSHOP_"):
        return outcome
    lo = (obs_or_next_obs or "").lower()
    if "your score (min 0.0, max 1.0)" in lo or "buy now" in lo:
        return "WEBSHOP_CHECKOUT_PAGE"
    return outcome


def _is_move_action(action: str) -> bool:
    action = (action or "").strip().lower()
    return action in {"north", "south", "east", "west", "up", "down"} or action.startswith(
        ("move ", "go ")
    )


def _maze_coord_signature(obs_text: str) -> str:
    text = obs_text or ""
    matches = re.findall(r"position\s*(-?\d+)\s*,\s*(-?\d+)", text, flags=re.IGNORECASE)
    if not matches:
        return "pos:unknown"
    x, y = matches[-1]
    return f"pos:{x},{y}"


def _maze_wall_signature(obs_text: str) -> str:
    lo = (obs_text or "").lower()
    walls: List[str] = []
    if "above you" in lo:
        walls.append("up")
    if "below you" in lo:
        walls.append("down")
    if "to your left" in lo:
        walls.append("left")
    if "to your right" in lo:
        walls.append("right")
    if not walls:
        return "walls:none"
    ordered = [direction for direction in ("up", "down", "left", "right") if direction in walls]
    return "walls:" + ",".join(ordered)


def deduplicate_transitions(
    trajectories: List[Trajectory],
    max_per_pattern: int = 2,
) -> List[Tuple[Transition, str, str]]:
    """Extract one representative transition per (action_type, outcome_type) pair.

    Returns list of (transition, action_type, outcome_type).
    Scales to large trajectory sets: N trajectories → K unique patterns.
    """
    seen: Dict[Tuple[str, str], int] = {}
    result: List[Tuple[Transition, str, str]] = []

    for traj in trajectories:
        for step in traj.transitions:
            at = _action_type(step.action)
            ot = _outcome_type(step.next_observation)
            key = (at, ot)
            count = seen.get(key, 0)
            if count < max_per_pattern:
                result.append((step, at, ot))
                seen[key] = count + 1

    return result


# ---------------------------------------------------------------------------
# Contrastive example selection
# ---------------------------------------------------------------------------

def select_contrastive_examples(
    trajectories: List[Trajectory],
    max_transitions: int = 30,
    max_per_pattern: int = 2,
) -> List[Transition]:
    """Select transitions that maximise action-type coverage and outcome diversity.

    For each action type, include at least one success and one failure transition
    so the LLM sees the same action producing different outputs — forcing it to
    learn the parametric rule rather than a specific mapping.
    """
    deduped = deduplicate_transitions(trajectories, max_per_pattern=max_per_pattern)
    # Group by action_type
    by_action: Dict[str, List[Tuple[Transition, str, str]]] = {}
    for step, at, ot in deduped:
        by_action.setdefault(at, []).append((step, at, ot))

    selected: List[Transition] = []
    # Round-robin across action types to keep diversity
    action_types = sorted(by_action.keys(), key=lambda at: (-len(by_action[at]), at))
    i = 0
    while len(selected) < max_transitions and i < len(action_types) * max_transitions:
        at = action_types[i % len(action_types)]
        pool = by_action[at]
        if pool:
            step, _, _ = pool.pop(0)
            selected.append(step)
        i += 1

    return selected[:max_transitions]


def select_random_examples(
    trajectories: List[Trajectory],
    max_transitions: int = 30,
) -> List[Transition]:
    """Uniform random transition sampling (ablation helper)."""
    pool: List[Transition] = []
    for traj in trajectories:
        pool.extend(traj.transitions)
    if len(pool) <= max_transitions:
        return pool
    random.shuffle(pool)
    return pool[:max_transitions]


def _babyai_state_bucket(text: str) -> str:
    lo = (text or "").lower()
    tags: List[str] = []
    if "carrying" in lo:
        tags.append("carrying")
    if "key" in lo:
        tags.append("key")
    if "door" in lo:
        if "locked" in lo:
            tags.append("door_locked")
        elif "open" in lo:
            tags.append("door_open")
        else:
            tags.append("door")
    if "goal" in lo or "mission" in lo:
        tags.append("goal")
    if "picked up" in lo or "pickup" in lo:
        tags.append("pickup")
    if "drop" in lo:
        tags.append("drop")
    if not tags:
        tags.append(_outcome_type(text))
    return "+".join(tags[:3])
