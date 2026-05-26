"""World-model-guided agent for AgentGym environments.

Interacts LIVE with the AgentGym environment servers (HTTP). Uses the
world model to score/filter candidate actions before executing them.

Architecture
------------
  BaseAgent         — pure ReAct agent (baseline), drives the env client directly
  WorldModelAgent   — wraps BaseAgent, inserts world-model scoring each step

Both agents work with the agentenv *EnvClient objects (AlfWorldEnvClient,
MazeEnvClient, etc.) which talk to the running HTTP servers.

Decision loop (WorldModelAgent, per step)
-----------------------------------------
  1. Update world model belief from the latest observation.
  2. Build candidate action list:
       a. World model's extract_valid_action_forms() templates
       b. Available actions reported by the environment (if any)
       c. One LLM-proposed action via ReAct prompt
  3. Predict next observation for each candidate via predict_belief + readout_observation.
  4. Prompt LLM to pick the best candidate given predictions + goal.
  5. Execute chosen action; update belief with actual next observation.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from patchworld.worldmodel_base import BaseWorldModel, GraphState
from patchworld.worldmodel_inducer import LLMClient
from patchworld.config import DEFAULT_SERVER_BASE


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_SYSTEM_REACT = (
    "Interact with a text-based environment to solve a task. "
    "At each step, respond in this EXACT format:\n"
    "Thought: <brief reasoning>\n"
    "Action: <the action to take>"
)

_USER_REACT = """\
Environment: {env}
Goal: {goal}

Recent history:
{history}

Current observation:
{obs}

What is your next action?"""

_USER_WM_SELECT = """\
Environment: {env}
Goal: {goal}

Current observation:
{obs}

The world model predicts these outcomes for each candidate action:
{candidates}

Pick the ONE best action from the candidates list to make progress toward the goal.
Respond:
Thought: <brief reasoning>
Action: <exact action string>"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_action(text: str) -> str:
    """Pull the action string from an LLM response."""
    for line in text.splitlines():
        stripped = line.strip()
        if re.match(r"(?i)^action\s*:", stripped):
            return stripped.split(":", 1)[1].strip()
    # Fallback: last non-empty line
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return lines[-1] if lines else text.strip()


def _extract_goal(first_obs: str) -> str:
    """Heuristically extract the task goal from the first observation."""
    for marker in ["Your task is to", "You need to", "Task:", "Goal:", "Find"]:
        if marker.lower() in first_obs.lower():
            idx = first_obs.lower().index(marker.lower())
            return first_obs[idx: idx + 250].split("\n")[0].strip()
    return first_obs[:200].split("\n")[0].strip()


def _belief_summary(belief: Any, max_chars: int = 350) -> str:
    if isinstance(belief, GraphState):
        parts = []
        meta = belief.metadata
        if meta.get("agent_location"):
            parts.append(f"location={meta['agent_location']}")
        if meta.get("inventory"):
            parts.append(f"inventory={list(meta['inventory'])}")
        if meta.get("goal"):
            parts.append(f"goal={str(meta['goal'])[:100]}")
        return (", ".join(parts) or "(empty graph)") [:max_chars]
    if isinstance(belief, dict):
        return json.dumps(belief, default=str)[:max_chars]
    return str(belief)[:max_chars]


def _make_env_client(env_name: str, env_server_base: Optional[str] = None, data_len: int = 200):
    """Instantiate the agentenv *EnvClient for the given environment."""
    from agentenv.envs import (
        AlfWorldTask, BabyAITask, MazeTask, WordleTask,
        TextCraftTask, WebshopTask, SciworldTask,
    )
    task_classes = {
        "alfworld":  AlfWorldTask,
        "babyai":    BabyAITask,
        "maze":      MazeTask,
        "wordle":    WordleTask,
        "textcraft": TextCraftTask,
        "webshop":   WebshopTask,
        "sciworld":  SciworldTask,
    }
    cls = task_classes.get(env_name.lower())
    if cls is None:
        raise ValueError(
            f"[world_model_agent] Unsupported env '{env_name}'. "
            f"Supported: {list(task_classes)}"
        )
    server_base = env_server_base or DEFAULT_SERVER_BASE.get(env_name)
    if not server_base:
        raise ValueError(f"[world_model_agent] No server_base for '{env_name}'.")

    task = cls(
        client_args={"env_server_base": server_base, "data_len": max(1, int(data_len)), "timeout": 2400},
        n_clients=1,
    )
    return task, task.clients[0]


def _get_available_actions(client) -> List[str]:
    """Extract the environment's available actions list from the client, if exposed."""
    info = getattr(client, "info", None)
    if isinstance(info, dict):
        avail = info.get("available_actions", [])
        if avail:
            return list(avail)
    return []


def _observe(client) -> str:
    """Get the current observation text from the client."""
    try:
        return client.observe()
    except Exception:
        info = getattr(client, "info", {}) or {}
        return str(info.get("observation", ""))


# ---------------------------------------------------------------------------
# BaseAgent — pure ReAct, drives the env client directly
# ---------------------------------------------------------------------------

class BaseAgent:
    """ReAct agent with no world model (baseline)."""

    MAX_HISTORY = 5  # steps of (obs, action) kept in the prompt

    def __init__(
        self,
        env_name: str,
        llm: LLMClient,
        max_steps: int = 30,
    ):
        self.env_name = env_name
        self.llm = llm
        self.max_steps = max_steps
        self._history: List[Tuple[str, str]] = []
        self._goal: str = ""
        self._step: int = 0

    def reset(self, first_obs: str) -> None:
        self._history = []
        self._goal = _extract_goal(first_obs)
        self._step = 0

    def act(self, obs: str, available_actions: Optional[List[str]] = None) -> str:
        self._step += 1
        history_text = "\n".join(
            f"[{i+1}] Obs: {o[:150]}\n     Act: {a}"
            for i, (o, a) in enumerate(self._history[-self.MAX_HISTORY:])
        )
        obs_text = obs
        if available_actions:
            obs_text += f"\nAvailable actions: {', '.join(available_actions[:20])}"

        prompt = _USER_REACT.format(
            env=self.env_name,
            goal=self._goal,
            history=history_text or "(none)",
            obs=obs_text[:700],
        )
        try:
            messages = [
                {"role": "system", "content": _SYSTEM_REACT},
                {"role": "user",   "content": prompt},
            ]
            resp = self.llm.client.chat.completions.create(
                model=self.llm.model,
                messages=messages,
                temperature=0.3,
            )
            response = resp.choices[0].message.content or ""
            action = _extract_action(response)
        except Exception:
            action = "look"

        self._history.append((obs, action))
        return action

    def name(self) -> str:
        return "react-baseline"


# ---------------------------------------------------------------------------
# WorldModelAgent — inserts world-model scoring between perception and action
# ---------------------------------------------------------------------------

class WorldModelAgent(BaseAgent):
    """Agent that uses a world model to score/filter candidate actions."""

    MAX_CANDIDATES = 8
    MIN_WM_CANDIDATES = 2  # only do WM scoring if ≥2 candidates

    def __init__(
        self,
        env_name: str,
        llm: LLMClient,
        world_model: BaseWorldModel,
        max_steps: int = 30,
    ):
        super().__init__(env_name, llm, max_steps)
        self.world_model = world_model
        self._belief: Any = None

    def reset(self, first_obs: str) -> None:
        super().reset(first_obs)
        try:
            self._belief = self.world_model.init_belief_from_observation(first_obs)
        except Exception:
            self._belief = self.world_model.init_belief()

    def act(self, obs: str, available_actions: Optional[List[str]] = None) -> str:
        self._step += 1

        # Update belief with the incoming observation
        try:
            self._belief = self.world_model.correct_belief(self._belief, obs)
        except Exception:
            pass

        # Gather candidate actions
        candidates = self._build_candidates(obs, available_actions)

        if len(candidates) < self.MIN_WM_CANDIDATES:
            # Fall back to pure ReAct when there are almost no candidates
            action = self._react_action(obs, available_actions)
            self._history.append((obs, action))
            return action

        # Predict next obs for each candidate
        predictions = self._predict_candidates(candidates)

        # Ask LLM to select the best candidate given predictions
        action = self._select(obs, predictions)

        self._history.append((obs, action))
        return action

    def name(self) -> str:
        return "worldmodel-agent"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_candidates(
        self,
        obs: str,
        available_actions: Optional[List[str]],
    ) -> List[str]:
        candidates: List[str] = []

        # 1. Environment's own available-actions list (ground truth from server)
        if available_actions:
            candidates.extend(available_actions)

        # 2. World model's generic action templates
        try:
            forms = self.world_model.extract_valid_action_forms()
            for templates in forms.values():
                candidates.extend(templates)
        except Exception:
            pass

        # 3. One LLM-proposed action
        llm_action = self._react_action(obs, available_actions)
        if llm_action and llm_action not in candidates:
            candidates.insert(0, llm_action)

        # Deduplicate, preserve order
        seen: set = set()
        unique: List[str] = []
        for c in candidates:
            if c and c not in seen:
                seen.add(c)
                unique.append(c)

        return unique[: self.MAX_CANDIDATES]

    def _react_action(
        self,
        obs: str,
        available_actions: Optional[List[str]],
    ) -> str:
        history_text = "\n".join(
            f"[{i+1}] Obs: {o[:100]}\n     Act: {a}"
            for i, (o, a) in enumerate(self._history[-3:])
        )
        obs_text = obs
        if available_actions:
            obs_text += f"\nAvailable actions: {', '.join(available_actions[:20])}"
        prompt = _USER_REACT.format(
            env=self.env_name,
            goal=self._goal,
            history=history_text or "(none)",
            obs=obs_text[:600],
        )
        try:
            messages = [
                {"role": "system", "content": _SYSTEM_REACT},
                {"role": "user",   "content": prompt},
            ]
            resp = self.llm.client.chat.completions.create(
                model=self.llm.model,
                messages=messages,
                temperature=0.3,
            )
            return _extract_action(resp.choices[0].message.content or "")
        except Exception:
            return "look"

    def _predict_candidates(
        self, candidates: List[str]
    ) -> List[Tuple[str, str]]:
        out: List[Tuple[str, str]] = []
        for action in candidates:
            pred = ""
            try:
                belief_prior = self.world_model.predict_belief(self._belief, action)
                pred = self.world_model.readout_observation(belief_prior, action)
            except Exception:
                pass
            out.append((action, pred or "(no prediction)"))
        return out

    def _select(
        self,
        obs: str,
        predictions: List[Tuple[str, str]],
    ) -> str:
        candidates_text = "\n".join(
            f"  [{i+1}] ACTION: {act}\n"
            f"       PREDICTED NEXT OBS: {pred[:200]}"
            for i, (act, pred) in enumerate(predictions)
        )

        prompt = _USER_WM_SELECT.format(
            env=self.env_name,
            goal=self._goal,
            obs=obs[:500],
            candidates=candidates_text,
        )
        try:
            messages = [
                {"role": "system", "content": _SYSTEM_REACT},
                {"role": "user",   "content": prompt},
            ]
            resp = self.llm.client.chat.completions.create(
                model=self.llm.model,
                messages=messages,
                temperature=0.2,
            )
            chosen = _extract_action(resp.choices[0].message.content or "")

            # Verify it matches a candidate; if not, fuzzy-match
            candidate_actions = [a for a, _ in predictions]
            if chosen in candidate_actions:
                return chosen
            for ca in candidate_actions:
                if chosen.lower() in ca.lower() or ca.lower() in chosen.lower():
                    return ca
            # Fall back to first candidate
            return candidate_actions[0] if candidate_actions else chosen
        except Exception:
            return predictions[0][0] if predictions else "look"


# ---------------------------------------------------------------------------
# Episode runner — talks directly to the agentenv client
# ---------------------------------------------------------------------------

def run_episode(
    agent: BaseAgent,
    client,                 # agentenv *EnvClient instance
    task_idx: int,
    max_steps: int = 30,
    verbose: bool = False,
) -> Dict:
    """Run one episode against a live environment server.

    Args:
        agent:     BaseAgent or WorldModelAgent instance.
        client:    agentenv env client (AlfWorldEnvClient, MazeEnvClient, etc.)
        task_idx:  index into the environment's task list.
        max_steps: step budget.
        verbose:   print each (obs, action, reward) line.

    Returns:
        dict with episode stats.
    """
    client.reset(task_idx)
    obs = _observe(client)
    agent.reset(obs)

    total_reward = 0.0
    done = False
    steps = 0

    while not done and steps < max_steps:
        avail = _get_available_actions(client)

        if verbose:
            print(f"  [step {steps+1}] obs: {obs[:120].replace(chr(10), ' ')}")

        action = agent.act(obs, available_actions=avail if avail else None)

        if verbose:
            print(f"           act: {action}")

        try:
            step_out = client.step(action)
            obs = step_out.state
            reward = float(step_out.reward)
            done = bool(step_out.done)
        except Exception as e:
            if verbose:
                print(f"           [step error] {e}")
            break

        total_reward += reward
        steps += 1

    success = total_reward > 0 or (done and total_reward >= 0)

    if verbose:
        mark = "SUCCESS" if success else "FAIL"
        print(f"  [{mark}] steps={steps} total_reward={total_reward:.3f}")

    return {
        "task_idx": task_idx,
        "agent": agent.name(),
        "steps": steps,
        "total_reward": round(total_reward, 4),
        "success": success,
        "done": done,
    }
