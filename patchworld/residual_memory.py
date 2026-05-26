"""Train-only episodic residual memory for executable world models.

The wrapper keeps the generated symbolic model as the primary dynamics model.
It only overrides readout when the current observation/action pair matches a
high-confidence transition observed in the training trajectories.
"""

from __future__ import annotations

from collections import Counter
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

from patchworld.worldmodel_data import Trajectory


def _normalize_text(text: Any) -> str:
    return " ".join(str(text or "").strip().lower().split())


def _transition_key(observation: str, action: str) -> Tuple[str, str]:
    return (_normalize_text(observation), _normalize_text(action))


def _split_webshop_sections(observation: str) -> List[str]:
    return [part.strip() for part in str(observation or "").split("[SEP]") if part.strip()]


def _join_webshop_sections(parts: Iterable[str]) -> str:
    return " [SEP] ".join(str(part).strip() for part in parts if str(part).strip())


def _webshop_instruction(parts: List[str]) -> str:
    for idx, part in enumerate(parts):
        if part.lower() == "instruction:" and idx + 1 < len(parts):
            return parts[idx + 1]
    return ""


def _webshop_product_from_parts(parts: List[str], asin: str) -> Dict[str, str]:
    asin_upper = asin.strip().upper()
    for idx, part in enumerate(parts):
        if part.strip().upper() != asin_upper:
            continue
        product = {"asin": asin_upper, "title": "", "price": ""}
        if idx + 1 < len(parts):
            product["title"] = parts[idx + 1]
        for candidate in parts[idx + 2 : idx + 5]:
            if candidate.startswith("$") or candidate.lower().startswith("price:"):
                product["price"] = candidate.replace("Price:", "").strip()
                break
        return product
    return {"asin": asin_upper, "title": "", "price": ""}


def _webshop_current_asin(parts: List[str]) -> str:
    for part in parts:
        token = part.strip().upper()
        if re.fullmatch(r"B0[A-Z0-9]{8}", token):
            return token
    return ""


def _webshop_page_number(parts: List[str]) -> int:
    for part in parts:
        match = re.search(r"\bPage\s+(\d+)\b", part, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    return 1


def _webshop_heuristic_next_observation(observation: str, action: str) -> Optional[str]:
    """Action-conditioned WebShop fallback for unseen observation/action keys.

    It intentionally preserves concrete text already visible in the current page
    and only changes structural fields that WebShop actions determine locally.
    """
    parts = _split_webshop_sections(observation)
    if not parts:
        return None
    instruction = _webshop_instruction(parts)
    action_text = str(action or "").strip()
    action_lower = action_text.lower()

    if action_lower.startswith("search[") or action_lower == "click[search]":
        return _join_webshop_sections(
            ["Instruction:", instruction, "Back to Search", "Page 1", "Next >"]
        )

    if action_lower == "click[back to search]":
        return _join_webshop_sections(["WebShop", "Instruction:", instruction, "Search"])

    if action_lower == "click[next >]" or action_lower == "next":
        page = _webshop_page_number(parts) + 1
        updated: List[str] = []
        replaced = False
        for part in parts:
            if re.search(r"\bPage\s+\d+\b", part, flags=re.IGNORECASE):
                updated.append(re.sub(r"\bPage\s+\d+\b", f"Page {page}", part, flags=re.IGNORECASE))
                replaced = True
            else:
                updated.append(part)
        if not replaced:
            updated.extend(["< Prev", f"Page {page}", "Next >"])
        elif "< Prev" not in updated:
            page_idx = next(
                (
                    idx
                    for idx, part in enumerate(updated)
                    if re.search(r"\bPage\s+\d+\b", part, flags=re.IGNORECASE)
                ),
                0,
            )
            updated.insert(page_idx, "< Prev")
        return _join_webshop_sections(updated)

    if action_lower == "click[< prev]" or action_lower == "back":
        if any(part.lower() in {"description", "features", "reviews", "buy now"} for part in parts):
            return observation
        page = max(1, _webshop_page_number(parts) - 1)
        updated = []
        for part in parts:
            if re.search(r"\bPage\s+\d+\b", part, flags=re.IGNORECASE):
                updated.append(re.sub(r"\bPage\s+\d+\b", f"Page {page}", part, flags=re.IGNORECASE))
            elif page == 1 and part == "< Prev":
                continue
            else:
                updated.append(part)
        return _join_webshop_sections(updated)

    asin_match = re.fullmatch(r"click\[(b0[a-z0-9]{8})\]", action_lower)
    if asin_match:
        asin = asin_match.group(1).upper()
        product = _webshop_product_from_parts(parts, asin)
        rendered = ["Instruction:", instruction, "Back to Search", "< Prev", asin]
        if product.get("title"):
            rendered.append(product["title"])
        if product.get("price"):
            rendered.append(f"Price: {product['price']}")
        rendered.extend(["Description", "Features", "Reviews", "Buy Now"])
        return _join_webshop_sections(rendered)

    if action_lower in {"click[description]", "click[features]", "click[reviews]"}:
        return observation

    if action_lower == "click[buy now]":
        asin = _webshop_current_asin(parts)
        return _join_webshop_sections(
            [
                "Thank you for shopping with us!",
                "Your code:",
                "None",
                "(Paste it in your MTurk interface.)",
                "Purchased",
                "asin",
                asin or "None",
                "options",
                "{}",
                "Reward",
                "Your score (min 0.0, max 1.0)",
                "1.0",
            ]
        )

    if action_lower.startswith("click["):
        # Filter/option clicks usually keep the user on a concrete product or
        # option page. Preserve current page text rather than hallucinating.
        return observation

    return None


class EpisodicResidualWorldModel:
    """Semi-parametric wrapper: symbolic dynamics plus train-only residuals.

    The memory stores empirical next-observation residuals keyed by normalized
    current observation and action. Conflicting keys are used only when the most
    common target clears ``min_confidence``.
    """

    def __init__(
        self,
        base_model: Any,
        train_trajectories: Iterable[Trajectory],
        *,
        env_name: str = "",
        max_entries: int = 0,
        min_count: int = 1,
        min_confidence: float = 1.0,
    ) -> None:
        self.base_model = base_model
        self.env_name = env_name
        self.max_entries = max(0, int(max_entries or 0))
        self.min_count = max(1, int(min_count or 1))
        self.min_confidence = min(1.0, max(0.0, float(min_confidence)))
        self._memory: Dict[Tuple[str, str], str] = {}
        self._memory_confidence: Dict[Tuple[str, str], float] = {}
        self._last_obs_by_belief_id: Dict[int, str] = {}
        self._residual_by_prior_id: Dict[int, str] = {}
        self._stats = self._build_memory(list(train_trajectories))

    @property
    def residual_stats(self) -> Dict[str, Any]:
        return dict(self._stats)

    def _build_memory(self, trajectories: List[Trajectory]) -> Dict[str, Any]:
        buckets: Dict[Tuple[str, str], Counter[str]] = {}
        total_transitions = 0
        for traj in trajectories:
            for step in traj.transitions:
                total_transitions += 1
                key = _transition_key(step.observation, step.action)
                buckets.setdefault(key, Counter())[step.next_observation] += 1

        candidates: List[Tuple[int, Tuple[str, str], str, float, int]] = []
        conflict_keys = 0
        for key, counter in buckets.items():
            total_for_key = sum(counter.values())
            if len(counter) > 1:
                conflict_keys += 1
            next_obs, count = counter.most_common(1)[0]
            confidence = count / float(total_for_key or 1)
            if count >= self.min_count and confidence >= self.min_confidence:
                candidates.append((count, key, next_obs, confidence, total_for_key))

        candidates.sort(key=lambda item: (-item[0], -item[3], item[1]))
        if self.max_entries > 0:
            candidates = candidates[: self.max_entries]

        for _count, key, next_obs, confidence, _total_for_key in candidates:
            self._memory[key] = next_obs
            self._memory_confidence[key] = confidence

        return {
            "env": self.env_name,
            "train_trajectories": len(trajectories),
            "train_transitions": total_transitions,
            "raw_keys": len(buckets),
            "conflict_keys": conflict_keys,
            "entries": len(self._memory),
            "max_entries": self.max_entries,
            "min_count": self.min_count,
            "min_confidence": self.min_confidence,
            "key": "normalized_observation_action",
        }

    def _lookup(self, observation: Optional[str], action: str) -> Optional[str]:
        if not observation:
            return None
        return self._memory.get(_transition_key(observation, action))

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base_model, name)

    def init_belief(self) -> Any:
        return self.base_model.init_belief()

    def init_belief_from_observation(self, obs_text: str) -> Any:
        if hasattr(self.base_model, "init_belief_from_observation"):
            belief = self.base_model.init_belief_from_observation(obs_text)
        else:
            belief = self.base_model.init_belief()
            corrected = self.base_model.correct_belief(belief, obs_text)
            belief = belief if corrected is None else corrected
        self._last_obs_by_belief_id[id(belief)] = obs_text
        return belief

    def parse_observation(self, obs_text: str) -> Any:
        return self.base_model.parse_observation(obs_text)

    def correct_belief(self, belief_prior: Any, obs_text: str) -> Any:
        corrected = self.base_model.correct_belief(belief_prior, obs_text)
        belief = belief_prior if corrected is None else corrected
        self._last_obs_by_belief_id[id(belief)] = obs_text
        return belief

    def predict_belief(self, belief: Any, action: str) -> Any:
        current_obs = self._last_obs_by_belief_id.get(id(belief))
        residual_obs = self._lookup(current_obs, action)
        if residual_obs is None and self.env_name == "webshop" and current_obs:
            residual_obs = _webshop_heuristic_next_observation(current_obs, action)
        try:
            prior = self.base_model.predict_belief(belief, action)
        except Exception:
            prior = belief
        if residual_obs is not None:
            self._residual_by_prior_id[id(prior)] = residual_obs
        return prior

    def readout_observation(self, belief: Any, action: str = "") -> str:
        residual_obs = self._residual_by_prior_id.pop(id(belief), None)
        if residual_obs is not None:
            return residual_obs
        return self.base_model.readout_observation(belief, action) or ""

    def predict_observation(self, belief: Any, action: str) -> str:
        prior = self.predict_belief(belief, action)
        return self.readout_observation(prior, action)

    def transition(self, belief: Any, action: str) -> Any:
        return self.predict_belief(belief, action)

    def predict_next(self, state: Any, action: str) -> Any:
        return self.predict_belief(state, action)
