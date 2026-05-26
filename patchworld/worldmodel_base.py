from dataclasses import dataclass, field
import re
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


@dataclass
class GraphState:
    """Belief state for spatial environments (navigation, manipulation).

    nodes: entity_id -> attribute dict (e.g. {"type": "room", "visited": True})
    edges: set of (src_id, relation, dst_id) triples — supports both .add() and iteration
    metadata: task-level info (goal, inventory, agent_location, etc.)
    """

    nodes: Dict[str, Dict] = field(default_factory=dict)
    edges: Set[Tuple[str, str, str]] = field(default_factory=set)
    metadata: Dict = field(default_factory=dict)


@dataclass(frozen=True)
class BeliefHypothesis:
    """A weighted latent-world hypothesis maintained under partial observability."""

    hypothesis_id: str
    description: str = ""
    weight: float = 1.0
    payload: Dict[str, Any] = field(default_factory=dict)


MAZE_DIRECTIONS: Tuple[str, ...] = ("up", "down", "left", "right")


@dataclass(frozen=True)
class MazeState:
    """Canonical explicit state for maze-style navigation tasks."""

    agent_x: Optional[int] = None
    agent_y: Optional[int] = None
    goal_x: Optional[int] = None
    goal_y: Optional[int] = None
    local_walls: Tuple[str, ...] = ()
    status: str = ""
    obs_text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "agent_x": self.agent_x,
            "agent_y": self.agent_y,
            "goal_x": self.goal_x,
            "goal_y": self.goal_y,
            "local_walls": self.local_walls,
        }
        if self.status:
            data["status"] = self.status
        if self.obs_text:
            data["obs_text"] = self.obs_text
        return data


def _maze_wall_direction(value: Any) -> Optional[str]:
    if isinstance(value, str):
        direction = value.strip().lower()
        aliases = {
            "above": "up",
            "above you": "up",
            "up": "up",
            "below": "down",
            "below you": "down",
            "down": "down",
            "left": "left",
            "to your left": "left",
            "right": "right",
            "to your right": "right",
        }
        return aliases.get(direction)
    if isinstance(value, tuple) and value:
        tail = value[-1]
        if isinstance(tail, str):
            return _maze_wall_direction(tail)
    return None


def canonicalize_maze_walls(value: Any) -> Tuple[str, ...]:
    walls: Set[str] = set()
    if isinstance(value, dict):
        iterable = value.keys()
    elif isinstance(value, (list, tuple, set, frozenset)):
        iterable = value
    elif value is None:
        iterable = ()
    else:
        iterable = (value,)
    for item in iterable:
        direction = _maze_wall_direction(item)
        if direction in MAZE_DIRECTIONS:
            walls.add(direction)
    return tuple(direction for direction in MAZE_DIRECTIONS if direction in walls)


def coerce_maze_state(value: Any) -> Optional[MazeState]:
    """Best-effort conversion from parsed dicts or predictions to MazeState."""
    if isinstance(value, MazeState):
        return value
    if not isinstance(value, dict):
        return None

    status = str(value.get("status", "") or "").strip().lower()
    if status == "success":
        return MazeState(status="success", obs_text=str(value.get("obs_text", "") or ""))

    if str(value.get("raw_text", "") or "").strip() == "Success":
        return MazeState(status="success", obs_text="Success")

    keys = set(value.keys())
    if not ({"agent_x", "agent_y"} & keys):
        return None

    def _as_int(name: str) -> Optional[int]:
        raw = value.get(name)
        if raw is None:
            return None
        try:
            return int(raw)
        except (TypeError, ValueError):
            return None

    return MazeState(
        agent_x=_as_int("agent_x"),
        agent_y=_as_int("agent_y"),
        goal_x=_as_int("goal_x"),
        goal_y=_as_int("goal_y"),
        local_walls=canonicalize_maze_walls(value.get("local_walls", value.get("walls"))),
        status=status,
        obs_text=str(value.get("obs_text", "") or value.get("raw_text", "") or ""),
    )


def render_maze_observation(state: MazeState) -> str:
    """Render the canonical maze state back into the environment's text format."""
    if state.status == "success":
        return "Success"
    if None in (state.agent_x, state.agent_y, state.goal_x, state.goal_y):
        return state.obs_text or ""
    base = (
        f"The goal is at position {state.goal_x}, {state.goal_y}. "
        f"Your current position is at position {state.agent_x}, {state.agent_y}."
    )
    if not state.local_walls:
        return base
    wall_text = {
        "up": "above you",
        "down": "below you",
        "left": "to your left",
        "right": "to your right",
    }
    rendered_walls = [wall_text[direction] for direction in state.local_walls]
    if len(rendered_walls) == 1:
        return f"{base} There is a wall {rendered_walls[0]}."
    if len(rendered_walls) == 2:
        joined = ", ".join(rendered_walls)
    else:
        joined = ", ".join(rendered_walls[:-1]) + f", {rendered_walls[-1]}"
    return f"{base} There are walls {joined}."


@dataclass(frozen=True)
class WordleState:
    guess_history: Tuple[str, ...] = ()
    feedback_history: Tuple[str, ...] = ()
    current_feedback: Tuple[str, ...] = ()
    candidate_count: Optional[int] = None
    status: str = ""


@dataclass(frozen=True)
class TextcraftState:
    inventory: Tuple[Tuple[str, int], ...] = ()
    goal_item: Optional[str] = None
    last_action_outcome: str = ""
    recipe_count: Optional[int] = None


@dataclass(frozen=True)
class WebshopState:
    page_type: str = ""
    current_page_number: Optional[int] = None
    selected_filters: Tuple[Tuple[str, str], ...] = ()
    asin: str = ""
    goal_completed: bool = False


def _normalize_wordle_guess(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    guess = " ".join(value.strip().lower().split())
    if re.fullmatch(r"[a-z](?: [a-z]){4}", guess):
        return guess
    compact = "".join(ch for ch in guess if ch.isalpha())
    if len(compact) == 5:
        return " ".join(compact)
    return None


def _normalize_wordle_feedback(value: Any) -> Optional[Tuple[str, ...]]:
    if isinstance(value, (list, tuple)):
        items = tuple(str(item).strip().lower() for item in value)
        if len(items) == 5 and all(item in {"b", "y", "g"} for item in items):
            return items
    if not isinstance(value, str):
        return None
    feedback = " ".join(value.strip().lower().split())
    if re.fullmatch(r"[byg](?: [byg]){4}", feedback):
        return tuple(feedback.split(" "))
    return None


def _normalize_mapping_items(mapping: Any) -> Tuple[Tuple[str, Any], ...]:
    if not isinstance(mapping, dict):
        return ()
    normalized: List[Tuple[str, Any]] = []
    for key, value in sorted(mapping.items(), key=lambda item: str(item[0]).strip().lower()):
        normalized.append((str(key).strip().lower(), value))
    return tuple(normalized)


def _webshop_sections(value: Dict[str, Any]) -> Tuple[str, ...]:
    raw_sections = value.get("sections")
    if isinstance(raw_sections, (list, tuple)):
        return tuple(str(section).strip() for section in raw_sections if str(section).strip())
    raw_text = str(
        value.get("raw_observation", "")
        or value.get("obs_text", "")
        or value.get("raw_text", "")
        or ""
    )
    if not raw_text:
        return ()
    return tuple(section.strip() for section in raw_text.split("[SEP]") if section.strip())


def _infer_webshop_page_type(value: Dict[str, Any], sections: Tuple[str, ...]) -> str:
    raw_page_type = str(
        value.get("page_type", "")
        or value.get("type", "")
        or value.get("page", "")
        or ""
    ).strip().lower()
    aliases = {
        "search_page": "search",
        "search": "search",
        "results_page": "results",
        "search_results": "results",
        "results": "results",
        "product_page": "product",
        "product_detail": "product",
        "product": "product",
        "filter_options": "filter_options",
        "purchase_complete": "purchase_complete",
        "thank_you": "purchase_complete",
    }
    if raw_page_type in aliases:
        return aliases[raw_page_type]

    section_set = {section.lower() for section in sections}
    if any("thank you for shopping with us" in section for section in section_set):
        return "purchase_complete"
    if "webshop" in section_set and "search" in section_set and "back to search" not in section_set:
        return "search"
    if {"description", "features", "reviews"} & section_set or "buy now" in section_set:
        return "product"
    if "back to search" in section_set or any(section.startswith("page ") for section in section_set):
        return "results"
    return raw_page_type


def _infer_webshop_page_number(value: Dict[str, Any], sections: Tuple[str, ...]) -> Optional[int]:
    raw_page = value.get("current_page_number", value.get("page_number"))
    if raw_page is not None:
        try:
            return int(raw_page)
        except (TypeError, ValueError):
            pass
    for section in sections:
        match = re.search(r"\bPage\s+(\d+)\b", section, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def _infer_webshop_asin(value: Dict[str, Any], sections: Tuple[str, ...], page_type: str = "") -> str:
    for raw in (
        value.get("asin"),
        value.get("active_asin"),
        value.get("purchased_item"),
        value.get("purchased_asin"),
    ):
        token = str(raw or "").strip().upper()
        if re.fullmatch(r"B0[A-Z0-9]{8}", token):
            return token
    product_details = value.get("product_details")
    if isinstance(product_details, dict):
        token = str(product_details.get("asin", "") or "").strip().upper()
        if re.fullmatch(r"B0[A-Z0-9]{8}", token):
            return token
    cart = value.get("cart", {})
    if isinstance(cart, dict):
        token = str(cart.get("asin", "") or "").strip().upper()
        if re.fullmatch(r"B0[A-Z0-9]{8}", token):
            return token
    if page_type not in {"product", "purchase_complete"}:
        return ""
    page_content = value.get("page_content", ())
    if isinstance(page_content, (list, tuple)):
        for item in page_content:
            token = str(item).strip().upper()
            if re.fullmatch(r"B0[A-Z0-9]{8}", token):
                return token
    for section in sections:
        token = section.strip().upper()
        if re.fullmatch(r"B0[A-Z0-9]{8}", token):
            return token
    return ""


def coerce_wordle_state(value: Any) -> Optional[WordleState]:
    if not isinstance(value, dict):
        return None
    guess_history_raw = value.get("guess_history", ())
    if isinstance(guess_history_raw, set):
        guess_history_iter: Iterable[Any] = sorted(guess_history_raw)
    elif isinstance(guess_history_raw, (list, tuple)):
        guess_history_iter = guess_history_raw
    else:
        guess_history_iter = ()
    guess_history = tuple(
        guess
        for guess in (_normalize_wordle_guess(item) for item in guess_history_iter)
        if guess is not None
    )

    feedback_history_raw = value.get("feedback_history", ())
    if isinstance(feedback_history_raw, set):
        feedback_history_iter: Iterable[Any] = sorted(feedback_history_raw)
    elif isinstance(feedback_history_raw, (list, tuple)):
        feedback_history_iter = feedback_history_raw
    else:
        feedback_history_iter = ()
    feedback_history = tuple(
        " ".join(feedback)
        for feedback in (_normalize_wordle_feedback(item) for item in feedback_history_iter)
        if feedback is not None
    )

    current_feedback = _normalize_wordle_feedback(
        value.get("feedback_letters")
        or value.get("current_feedback")
        or value.get("raw_text")
    ) or ()

    candidate_words = value.get("possible_words", value.get("candidate_words"))
    candidate_count: Optional[int] = None
    if isinstance(candidate_words, (set, list, tuple)):
        candidate_count = len(candidate_words)
    elif isinstance(candidate_words, dict):
        candidate_count = len(candidate_words.keys())

    status = ""
    if value.get("is_solved") is True or current_feedback == ("g", "g", "g", "g", "g"):
        status = "solved"
    elif value.get("is_invalid") is True or str(value.get("raw_text", "")).strip().lower() == "invalid word":
        status = "invalid"
    elif value.get("is_welcome") is True:
        status = "welcome"
    elif current_feedback:
        status = "feedback"

    if not any((guess_history, feedback_history, current_feedback, candidate_count is not None, status)):
        return None
    return WordleState(
        guess_history=guess_history,
        feedback_history=feedback_history,
        current_feedback=current_feedback,
        candidate_count=candidate_count,
        status=status,
    )


def coerce_textcraft_state(value: Any) -> Optional[TextcraftState]:
    if not isinstance(value, dict):
        return None
    inventory_raw = value.get("inventory", {})
    inventory: Tuple[Tuple[str, int], ...] = ()
    if isinstance(inventory_raw, dict):
        normalized_items: List[Tuple[str, int]] = []
        for key, raw_count in sorted(inventory_raw.items(), key=lambda item: str(item[0]).strip().lower()):
            try:
                count = int(raw_count)
            except (TypeError, ValueError):
                continue
            normalized_items.append((str(key).strip().lower(), count))
        inventory = tuple(normalized_items)
    goal_item = value.get("goal_item")
    if goal_item is not None:
        goal_item = str(goal_item).strip().lower() or None
    last_action_outcome = str(value.get("last_action_outcome", "") or "").strip().lower()
    crafting_recipes = value.get("crafting_recipes")
    recipe_count = len(crafting_recipes) if isinstance(crafting_recipes, (list, tuple, set)) else None
    if not any((inventory, goal_item, last_action_outcome, recipe_count is not None)):
        return None
    return TextcraftState(
        inventory=inventory,
        goal_item=goal_item,
        last_action_outcome=last_action_outcome,
        recipe_count=recipe_count,
    )


def coerce_webshop_state(value: Any) -> Optional[WebshopState]:
    if not isinstance(value, dict):
        return None
    sections = _webshop_sections(value)
    page_type = _infer_webshop_page_type(value, sections)
    current_page_number = _infer_webshop_page_number(value, sections)
    selected_filters = _normalize_mapping_items(value.get("selected_filters"))
    asin = _infer_webshop_asin(value, sections, page_type)
    goal_completed = bool(value.get("goal_completed", False) or page_type == "purchase_complete")
    if not any((page_type, current_page_number is not None, selected_filters, asin, goal_completed)):
        return None
    return WebshopState(
        page_type=page_type,
        current_page_number=current_page_number,
        selected_filters=tuple((key, str(val).strip().lower()) for key, val in selected_filters),
        asin=asin,
        goal_completed=goal_completed,
    )


class BaseWorldModel:
    """Interface that all generated world models must implement.

    This stays intentionally lightweight so that LLM-generated models can
    reasonably implement it in one pass.

    IMPORTANT: in most benchmark environments the observation is only partial
    evidence about the real world. A good model should therefore maintain a
    belief state over hidden variables, not treat the observation text as the
    full environment state.

    Spatial environments (alfworld, sciworld, babyai, maze, textcraft) should
    typically use GraphState or a structured dict as their belief state.
    API-style environments (webshop, webarena) can use a plain dict.
    """

    def parse_observation(self, obs_text: str) -> Dict:
        """Extract structured evidence from observation text.

        The parsed dict does not need to be the full latent environment state.
        It should capture directly observed constraints/signals that
        `correct_belief()` can use to maintain a richer hidden-state belief.
        """
        raise NotImplementedError

    def extract_valid_action_forms(self) -> Dict[str, List[str]]:
        """Return known action templates learned from trajectories."""
        raise NotImplementedError

    def init_belief(self) -> Any:
        """Return an empty belief state.

        Spatial models should return GraphState(); API models can return {}.
        """
        return {}

    def init_belief_from_observation(self, obs_text: str) -> Any:
        """Initialize belief directly from the first observation.

        This is a convenience wrapper for explicit predict/correct loops:
          b1 = init_belief_from_observation(o1)
        """
        belief = self.init_belief()
        return self.correct_belief(belief, obs_text)

    def predict_belief(self, belief: Any, action: str) -> Any:
        """Predict prior belief after applying action (no observation correction).

        Canonical predict step:
          b_next_prior = f(belief, action)
        """
        raise NotImplementedError

    def correct_belief(self, belief_prior: Any, obs_text: str) -> Any:
        """Correct prior belief using actual observation evidence.

        Canonical correction step:
          b_next = g(b_next_prior, obs_text)
        """
        raise NotImplementedError

    def readout_observation(self, belief: Any, action: str = "") -> str:
        """Render observation text implied by current belief.

        This is the canonical readout function used after predict/correct steps:
          o_hat = readout_observation(belief)
        """
        raise NotImplementedError

    # Backward-compatible wrappers for downstream callers.
    def transition(self, belief: Any, action: str) -> Any:
        return self.predict_belief(belief, action)

    def predict_next(self, state: Dict, action: str) -> Dict:
        return self.predict_belief(state, action)

    def predict_observation(self, belief: Any, action: str) -> str:
        belief_next_prior = self.predict_belief(belief, action)
        return self.readout_observation(belief_next_prior, action)


def _iter_belief_mappings(belief: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(belief, dict):
        yield belief
        for value in belief.values():
            yield from _iter_belief_mappings(value)
        return
    if isinstance(belief, (list, tuple, set, frozenset)):
        for item in belief:
            yield from _iter_belief_mappings(item)
        return
    metadata = getattr(belief, "metadata", None)
    if isinstance(metadata, dict):
        yield metadata
        for value in metadata.values():
            yield from _iter_belief_mappings(value)
    fields = getattr(belief, "__dict__", None)
    if isinstance(fields, dict):
        yield fields
        for value in fields.values():
            yield from _iter_belief_mappings(value)


def belief_diagnostics(belief: Any) -> Dict[str, Any]:
    """Return a lightweight summary of whether belief tracks hidden-state hypotheses."""
    diagnostics = {
        "candidate_count": 0,
        "hypothesis_count": 0,
        "latent_signal_count": 0,
        "resolved_status_count": 0,
        "tracks_latent_state": False,
        "signals": [],
    }
    latent_keys = {
        "possible_words",
        "candidate_words",
        "candidate_set",
        "candidate_probs",
        "candidate_probabilities",
        "hypotheses",
        "weighted_hypotheses",
        "belief_state",
        "belief_distribution",
        "unknown_objects",
        "unseen_locations",
        "unopened_containers",
        "unexamined_entities",
        "hidden_state",
        "selected_filters",
        "latents",
        "latent_variables",
        "frontier",
        "facts",
        "epistemic_facts",
        "epistemic_status",
        "posterior",
        "probabilities",
        "weights",
        "object_location_beliefs",
        "location_beliefs",
        "container_content_beliefs",
        "content_priors",
    }
    seen_mapping_ids: Set[int] = set()
    for mapping in _iter_belief_mappings(belief):
        mapping_id = id(mapping)
        if mapping_id in seen_mapping_ids:
            continue
        seen_mapping_ids.add(mapping_id)
        for key in latent_keys:
            if key not in mapping:
                continue
            value = mapping[key]
            diagnostics["signals"].append(key)
            diagnostics["latent_signal_count"] += 1
            diagnostics["tracks_latent_state"] = True
            if isinstance(value, (set, list, tuple, dict)):
                diagnostics["candidate_count"] += len(value)
            elif hasattr(value, "__dict__"):
                diagnostics["candidate_count"] += 1
            if key == "epistemic_status" and isinstance(value, dict):
                for status in value.values():
                    raw_status = getattr(status, "value", status)
                    if not isinstance(raw_status, str):
                        continue
                    normalized = raw_status.strip().lower()
                    if normalized and normalized not in {"unknown", "unobserved", "default", "none"}:
                        diagnostics["resolved_status_count"] += 1
        hypotheses = mapping.get("hypotheses")
        if isinstance(hypotheses, (list, tuple, set)):
            diagnostics["hypothesis_count"] += len(hypotheses)
            if hypotheses:
                diagnostics["tracks_latent_state"] = True
        elif isinstance(hypotheses, dict):
            diagnostics["hypothesis_count"] += len(hypotheses)
            if hypotheses:
                diagnostics["tracks_latent_state"] = True
    diagnostics["signals"] = sorted(set(diagnostics["signals"]))
    return diagnostics


def belief_supports_latent_inference(belief: Any, env_name: str = "") -> bool:
    """Heuristic check for whether a belief carries hidden-state support."""
    diagnostics = belief_diagnostics(belief)
    if diagnostics["tracks_latent_state"]:
        return True
    env = (env_name or "").strip().lower()
    if env == "maze":
        return True
    return False
