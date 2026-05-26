"""Evaluation metrics for induced world models.

Two primary metrics:
  - experience_fit:   how well the model replays *known* trajectories
                      (parse + transition/prediction consistency)
  - observation_fit:  how well the model predicts *unseen* next observations
                      (token-F1 between predicted and actual next obs text)

Use evaluate_world_model() as the main entry point.
"""

from __future__ import annotations

from datetime import datetime
import importlib
import re
from statistics import mean, pstdev
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from patchworld.worldmodel_base import (
    belief_diagnostics,
    belief_supports_latent_inference,
    coerce_maze_state,
    coerce_textcraft_state,
    coerce_webshop_state,
    coerce_wordle_state,
)
from patchworld.worldmodel_data import Trajectory
from patchworld.worldmodel_validator import load_model_from_code

_bleu_mod = None
try:
    _bleu_mod = importlib.import_module("nltk.translate.bleu_score")
except ModuleNotFoundError:
    _bleu_mod = None

SmoothingFunction = getattr(_bleu_mod, "SmoothingFunction", None)
sentence_bleu = getattr(_bleu_mod, "sentence_bleu", None)

STATE_FIELD_ALIASES: Dict[str, Tuple[str, ...]] = {
    "room": (
        "room",
        "agent_location",
        "location",
        "current_room",
        "current_location",
        "position",
    ),
    "objects": (
        "objects",
        "object_locations",
        "visible_objects",
        "room_objects",
        "items",
    ),
    "inventory": (
        "inventory",
        "carrying",
        "held_items",
        "held_objects",
        "bag",
    ),
}
SPECIALIZED_FIELD_NAMES: Tuple[str, ...] = (
    "agent_position",
    "goal_position",
    "wall_set",
    "terminal_status",
    "wordle_feedback",
    "wordle_status",
    "wordle_candidate_count",
    "textcraft_inventory",
    "textcraft_goal_item",
    "textcraft_last_action",
    "textcraft_recipe_count",
    "webshop_page_type",
    "webshop_page_number",
    "webshop_filters",
    "webshop_asin",
    "webshop_goal_completed",
)


def _token_f1(predicted: str, actual: str) -> float:
    """Compute unigram token overlap F1 between two strings."""
    pred_tokens = set(predicted.lower().split())
    actual_tokens = set(actual.lower().split())
    if not pred_tokens or not actual_tokens:
        return 0.0
    common = pred_tokens & actual_tokens
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(actual_tokens)
    return 2 * precision * recall / (precision + recall)


def _safe_bleu4(predicted: str, actual: str) -> float:
    """Sentence BLEU-4 with smoothing; returns 0 on unavailable deps/errors."""
    if not sentence_bleu or not SmoothingFunction:
        return 0.0
    pred_tokens = predicted.split()
    ref_tokens = actual.split()
    if not pred_tokens or not ref_tokens:
        return 0.0
    try:
        smoothing = SmoothingFunction().method1
        return float(
            sentence_bleu(
                [ref_tokens],
                pred_tokens,
                weights=(0.25, 0.25, 0.25, 0.25),
                smoothing_function=smoothing,
            )
        )
    except Exception:
        return 0.0


def _is_non_error_prediction(predicted_obs: str) -> bool:
    """Coverage counts non-empty and non-error-like predictions only."""
    if not predicted_obs or not predicted_obs.strip():
        return False
    lo = predicted_obs.lower()
    return not any(marker in lo for marker in ("traceback", "exception", "error:"))


def _infer_env_name(model: Any) -> str:
    cls_name = getattr(getattr(model, "__class__", None), "__name__", "") or ""
    lowered = cls_name.lower()
    for env_name in ("alfworld", "babyai", "maze", "sciworld", "textcraft", "webshop", "wordle"):
        if env_name in lowered:
            return env_name
    return ""


def _init_belief_from_first_observation(model: Any, first_obs: str) -> Any:
    belief = model.init_belief()
    corrected = model.correct_belief(belief, first_obs)
    return belief if corrected is None else corrected


def _predict_belief_prior(model: Any, belief: Any, action: str) -> Any:
    return model.predict_belief(belief, action)


def _readout_observation_from_belief(model: Any, belief: Any, action: str) -> str:
    return model.readout_observation(belief, action) or ""


def _correct_belief_with_observation(model: Any, belief_prior: Any, obs_text: str) -> Any:
    corrected = model.correct_belief(belief_prior, obs_text)
    return belief_prior if corrected is None else corrected


def _is_placeholder_prediction(predicted_obs: str, env_name: str = "") -> bool:
    text = (predicted_obs or "").strip().lower()
    if not text:
        return False
    generic_patterns = (
        "<asin>",
        "<title>",
        "<price>",
        "<rating>",
        "<filter_options>",
        "<product_details>",
        "<results_list>",
        "<description_content>",
        "<features_content>",
        "<reviews_content>",
        "[room_name]",
        "action effect not modeled.",
        "no known action matches that input",
        "sample product",
        "placeholder",
    )
    if any(pattern in text for pattern in generic_patterns):
        return True

    env = (env_name or "").strip().lower()
    if env == "webshop" and text.count("[sep]") >= 4 and re.search(r"<[^>]+>", text):
        return True
    if env in {"alfworld", "sciworld", "textcraft"} and text == "nothing happens.":
        return True
    if env == "textcraft" and "could not find enough items to craft item" in text:
        return True
    return False


def _canonical_key(key: Any) -> str:
    return str(key).strip().lower()


def _normalize_scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        return " ".join(value.strip().lower().split())
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value
    return str(value).strip().lower()


def _normalize_value(value: Any, _seen: Optional[set[int]] = None) -> Any:
    if _seen is None:
        _seen = set()

    is_container = isinstance(value, (dict, set, frozenset, list, tuple))
    obj_id = id(value)
    if is_container:
        if obj_id in _seen:
            # Prevent infinite recursion on self-referential beliefs/parsed structures.
            return "<cycle>"
        _seen.add(obj_id)

    try:
        if isinstance(value, dict):
            return {
                _canonical_key(k): _normalize_value(v, _seen)
                for k, v in sorted(value.items(), key=lambda item: _canonical_key(item[0]))
            }
        if isinstance(value, (set, frozenset, list, tuple)):
            normalized_items = [_normalize_value(item, _seen) for item in value]
            return tuple(sorted(normalized_items, key=lambda item: repr(item)))
        return _normalize_scalar(value)
    finally:
        if is_container:
            _seen.remove(obj_id)


def _first_present(parsed: Dict[str, Any], keys: Iterable[str]) -> Tuple[bool, Any]:
    for key in keys:
        if key in parsed:
            return True, _normalize_value(parsed[key])
        lowered = key.lower()
        for candidate, value in parsed.items():
            if _canonical_key(candidate) == lowered:
                return True, _normalize_value(value)
    return False, None


def extract_observation_state_fields(parsed: Dict[str, Any]) -> Dict[str, Any]:
    """Extract canonical observation fields used for state-level scoring."""
    if not isinstance(parsed, dict):
        return {}
    fields: Dict[str, Any] = {}
    for field_name, aliases in STATE_FIELD_ALIASES.items():
        present, value = _first_present(parsed, aliases)
        if present:
            fields[field_name] = value
    return fields


def normalize_parsed_observation(parsed: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(parsed, dict):
        return {}
    return {
        _canonical_key(key): _normalize_value(value)
        for key, value in sorted(parsed.items(), key=lambda item: _canonical_key(item[0]))
    }


def extract_maze_state_fields(parsed: Dict[str, Any]) -> Dict[str, Any]:
    maze_state = coerce_maze_state(parsed)
    if maze_state is None:
        return {}
    fields: Dict[str, Any] = {}
    if maze_state.agent_x is not None and maze_state.agent_y is not None:
        fields["agent_position"] = (maze_state.agent_x, maze_state.agent_y)
    if maze_state.goal_x is not None and maze_state.goal_y is not None:
        fields["goal_position"] = (maze_state.goal_x, maze_state.goal_y)
    if maze_state.local_walls or "local_walls" in parsed or "walls" in parsed:
        fields["wall_set"] = maze_state.local_walls
    if maze_state.status:
        fields["terminal_status"] = maze_state.status
    return fields


def extract_wordle_state_fields(parsed: Dict[str, Any]) -> Dict[str, Any]:
    wordle_state = coerce_wordle_state(parsed)
    if wordle_state is None:
        return {}
    fields: Dict[str, Any] = {}
    if wordle_state.current_feedback:
        fields["wordle_feedback"] = wordle_state.current_feedback
    if wordle_state.status:
        fields["wordle_status"] = wordle_state.status
    if wordle_state.candidate_count is not None:
        fields["wordle_candidate_count"] = wordle_state.candidate_count
    return fields


def extract_textcraft_state_fields(parsed: Dict[str, Any]) -> Dict[str, Any]:
    textcraft_state = coerce_textcraft_state(parsed)
    if textcraft_state is None:
        return {}
    fields: Dict[str, Any] = {}
    if textcraft_state.inventory:
        fields["textcraft_inventory"] = textcraft_state.inventory
    if textcraft_state.goal_item:
        fields["textcraft_goal_item"] = textcraft_state.goal_item
    if textcraft_state.last_action_outcome:
        fields["textcraft_last_action"] = textcraft_state.last_action_outcome
    if textcraft_state.recipe_count is not None:
        fields["textcraft_recipe_count"] = textcraft_state.recipe_count
    return fields


def extract_webshop_state_fields(parsed: Dict[str, Any]) -> Dict[str, Any]:
    webshop_state = coerce_webshop_state(parsed)
    if webshop_state is None:
        return {}
    fields: Dict[str, Any] = {}
    if webshop_state.page_type:
        fields["webshop_page_type"] = webshop_state.page_type
    if webshop_state.current_page_number is not None:
        fields["webshop_page_number"] = webshop_state.current_page_number
    if webshop_state.selected_filters:
        fields["webshop_filters"] = webshop_state.selected_filters
    if webshop_state.asin:
        fields["webshop_asin"] = webshop_state.asin
    fields["webshop_goal_completed"] = webshop_state.goal_completed
    return fields


def compare_parsed_observations(
    predicted: Dict[str, Any],
    actual: Dict[str, Any],
) -> Dict[str, Any]:
    """Compare parsed predicted and gold observations at state level."""
    pred_norm = normalize_parsed_observation(predicted)
    actual_norm = normalize_parsed_observation(actual)
    pred_specialized_fields: Dict[str, Any] = {}
    actual_specialized_fields: Dict[str, Any] = {}
    for extractor in (
        extract_maze_state_fields,
        extract_wordle_state_fields,
        extract_textcraft_state_fields,
        extract_webshop_state_fields,
    ):
        pred_specialized_fields.update(extractor(predicted))
        actual_specialized_fields.update(extractor(actual))
    pred_fields = extract_observation_state_fields(pred_norm)
    actual_fields = extract_observation_state_fields(actual_norm)
    if pred_specialized_fields or actual_specialized_fields:
        pred_fields = {**pred_fields, **pred_specialized_fields}
        actual_fields = {**actual_fields, **actual_specialized_fields}
        field_matches: Dict[str, Optional[bool]] = {}
        compared_fields = 0
        matched_fields = 0
        ordered_fields = ("room", "objects", "inventory") + SPECIALIZED_FIELD_NAMES
        for field_name in ordered_fields:
            pred_has = field_name in pred_fields
            actual_has = field_name in actual_fields
            if not pred_has and not actual_has:
                field_matches[field_name] = None
                continue
            compared_fields += 1
            match = pred_fields.get(field_name) == actual_fields.get(field_name)
            field_matches[field_name] = match
            if match:
                matched_fields += 1
        return {
            "state_match": matched_fields == compared_fields if compared_fields > 0 else False,
            "field_matches": field_matches,
            "compared_fields": compared_fields,
            "field_match_fraction": (matched_fields / compared_fields if compared_fields > 0 else 0.0),
            "predicted_fields": pred_fields,
            "actual_fields": actual_fields,
        }

    field_matches: Dict[str, Optional[bool]] = {}
    compared_fields = 0
    matched_fields = 0
    for field_name in ("room", "objects", "inventory"):
        pred_has = field_name in pred_fields
        actual_has = field_name in actual_fields
        if not pred_has and not actual_has:
            field_matches[field_name] = None
            continue
        compared_fields += 1
        match = pred_fields.get(field_name) == actual_fields.get(field_name)
        field_matches[field_name] = match
        if match:
            matched_fields += 1

    if compared_fields == 0:
        full_state_match = pred_norm == actual_norm
        field_matches["full_state"] = full_state_match
        compared_fields = 1
        matched_fields = 1 if full_state_match else 0

    return {
        "state_match": matched_fields == compared_fields,
        "field_matches": field_matches,
        "compared_fields": compared_fields,
        "field_match_fraction": (matched_fields / compared_fields if compared_fields > 0 else 0.0),
        "predicted_fields": pred_fields,
        "actual_fields": actual_fields,
    }


def compute_experience_fit(model: Any, trajectories: List[Trajectory]) -> Dict:
    """Measure how accurately the model replays known trajectories.

    Returns:
        parse_success_rate        – fraction of obs parsed without exception
        predict_success_rate      – fraction of transitions predicted without exception
        overall                   – mean of parse/predict rates
    """
    parse_total = parse_ok = 0
    predict_total = predict_ok = 0
    latent_belief_total = latent_belief_ok = 0

    for traj in trajectories:
        if not traj.transitions:
            continue
        belief = _init_belief_from_first_observation(model, traj.transitions[0].observation)
        state: Dict = {}
        env_name = (getattr(traj, "env", "") or "").strip().lower()

        for step in traj.transitions:
            next_parsed: Dict[str, Any] = {}
            # Parse observation — succeed if either parse_observation OR correction works
            parse_total += 1
            parse_succeeded = False
            try:
                parsed = model.parse_observation(step.observation)
                if isinstance(parsed, dict):
                    state.update(parsed)
                parse_succeeded = True
            except NotImplementedError:
                pass
            except Exception:
                pass
            try:
                belief = _correct_belief_with_observation(model, belief, step.observation)
                parse_succeeded = True
            except Exception:
                pass
            if env_name and env_name != "maze":
                latent_belief_total += 1
                if belief_supports_latent_inference(belief, env_name):
                    latent_belief_ok += 1
            if parse_succeeded:
                parse_ok += 1

            try:
                parsed_next_candidate = model.parse_observation(step.next_observation)
                if isinstance(parsed_next_candidate, dict):
                    next_parsed = parsed_next_candidate
            except Exception:
                next_parsed = {}

            current_maze = coerce_maze_state(state)
            next_maze = coerce_maze_state(next_parsed)
            is_maze_transition = current_maze is not None or next_maze is not None

            # Predict — succeed if either predict_next OR predict_observation works
            predict_total += 1
            predict_succeeded = False
            predicted_state: Any = None
            try:
                predicted_state = model.predict_belief(belief, step.action)
                predict_succeeded = True
            except NotImplementedError:
                pass
            except Exception:
                pass
            predicted_obs = ""
            try:
                predicted_obs = model.readout_observation(predicted_state, step.action) or ""
            except Exception:
                predicted_obs = ""
            if is_maze_transition:
                predict_succeeded = False
                if isinstance(predicted_state, dict) and next_parsed:
                    next_state_comparison = compare_parsed_observations(predicted_state, next_parsed)
                    state_ok = next_state_comparison["state_match"]
                else:
                    state_ok = False
                render_ok = False
                if predicted_obs:
                    try:
                        predicted_obs_parsed = model.parse_observation(predicted_obs)
                    except Exception:
                        predicted_obs_parsed = {}
                    if isinstance(predicted_obs_parsed, dict) and next_parsed:
                        render_ok = compare_parsed_observations(predicted_obs_parsed, next_parsed)["state_match"]
                predict_succeeded = state_ok and render_ok
            elif not predict_succeeded and predicted_obs:
                predict_succeeded = True
            if predict_succeeded:
                predict_ok += 1

            # Advance state with next observation
            if isinstance(next_parsed, dict):
                state.update(next_parsed)
            try:
                belief = _correct_belief_with_observation(model, belief, step.next_observation)
            except Exception:
                pass

    def _rate(ok: int, total: int) -> float:
        return ok / total if total > 0 else 0.0

    psr = _rate(parse_ok, parse_total)
    pred_sr = _rate(predict_ok, predict_total)

    return {
        "parse_success_rate": round(psr, 4),
        "predict_success_rate": round(pred_sr, 4),
        "latent_belief_tracking_rate": round(_rate(latent_belief_ok, latent_belief_total), 4),
        "overall": round((psr + pred_sr) / 2, 4),
        "n_trajectories": len(trajectories),
        "n_transitions": predict_total,
    }


def compute_observation_fit(
    model: Any,
    trajectories: List[Trajectory],
    *,
    progress_every: int = 0,
    progress_prefix: str = "",
    env_name: str = "",
) -> Dict:
    """Measure how well the model predicts next observation text.

    Calls predict_belief(belief, action) + readout_observation(prior, action) before each step.
    If the model returns a non-empty, non-error string, token-F1 against actual next_obs
    is recorded. If the model always returns '', coverage is 0 and mean_f1
    is 0.

    Note:
        Some baselines (e.g. WorldCoder/Poe-World) report "observation fit" as an
        *exact match* ratio on the predicted next observation text. To make
        metrics comparable across baselines, we also compute exact-match rates
        here (both on covered transitions and over all transitions).

    Args:
        progress_every: If > 0 and progress_prefix is set, print a line every N transitions.
        progress_prefix: Label for progress lines (e.g. \"[maze] neurosymbolic train\").

    Returns:
        mean_token_f1_pre_correction – primary metric: readout from prior belief b' (before correction)
        mean_token_f1_post_correction – auxiliary metric: readout after correcting with true next obs
        mean_token_f1            – average token-F1 across transitions with predictions
        mean_bleu4               – average sentence BLEU-4 across covered transitions
        exact_match_rate         – fraction of covered transitions with predicted_obs == actual_next_obs
        exact_match_rate_all     – fraction of all transitions with a correct exact-match prediction
        state_match_rate         – fraction of parsed covered transitions matching on key state fields
        state_match_rate_all     – fraction of all transitions matching on key state fields
        state_coverage           – fraction of transitions where both predicted and gold obs parsed
        room_match_rate          – fraction of comparable parsed transitions with matching room/location
        objects_match_rate       – fraction of comparable parsed transitions with matching objects
        inventory_match_rate     – fraction of comparable parsed transitions with matching inventory
        coverage                 – fraction of transitions where model gave a prediction
        post_correction_coverage – fraction of transitions where post-correction readout was available
        n_transitions            – total transitions evaluated
    """
    inferred_env_name = (env_name or _infer_env_name(model)).strip().lower()
    total = covered = 0
    f1_sum = 0.0
    bleu4_sum = 0.0
    post_corrected_covered = 0
    post_corrected_f1_sum = 0.0
    em_ok = 0
    em_ok_all = 0
    state_covered = 0
    state_match_ok = 0
    state_match_fraction_sum = 0.0
    covered_non_placeholder = 0
    non_placeholder_f1_sum = 0.0
    state_covered_non_placeholder = 0
    state_match_ok_non_placeholder = 0
    placeholder_violations = 0
    predicted_parse_ok = 0
    predicted_parse_attempts = 0
    field_stats = {
        "room": {"covered": 0, "ok": 0},
        "objects": {"covered": 0, "ok": 0},
        "inventory": {"covered": 0, "ok": 0},
        "agent_position": {"covered": 0, "ok": 0},
        "goal_position": {"covered": 0, "ok": 0},
        "wall_set": {"covered": 0, "ok": 0},
        "terminal_status": {"covered": 0, "ok": 0},
        "wordle_feedback": {"covered": 0, "ok": 0},
        "wordle_status": {"covered": 0, "ok": 0},
        "wordle_candidate_count": {"covered": 0, "ok": 0},
        "textcraft_inventory": {"covered": 0, "ok": 0},
        "textcraft_goal_item": {"covered": 0, "ok": 0},
        "textcraft_last_action": {"covered": 0, "ok": 0},
        "textcraft_recipe_count": {"covered": 0, "ok": 0},
        "webshop_page_type": {"covered": 0, "ok": 0},
        "webshop_page_number": {"covered": 0, "ok": 0},
        "webshop_filters": {"covered": 0, "ok": 0},
        "webshop_asin": {"covered": 0, "ok": 0},
        "webshop_goal_completed": {"covered": 0, "ok": 0},
    }
    total_transitions = sum(len(t.transitions) for t in trajectories)
    belief_candidate_count_sum = 0.0
    belief_candidate_count_steps = 0

    for traj in trajectories:
        if not traj.transitions:
            continue
        traj_env_name = (getattr(traj, "env", "") or inferred_env_name).strip().lower()
        belief = _init_belief_from_first_observation(model, traj.transitions[0].observation)

        for step in traj.transitions:
            total += 1
            if (
                progress_every > 0
                and progress_prefix
                and total_transitions > 0
                and total % progress_every == 0
            ):
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(
                    f"[{ts}] {progress_prefix}  observation_fit  "
                    f"transition {total}/{total_transitions}",
                    flush=True,
                )

            if traj_env_name:
                diagnostics = belief_diagnostics(belief)
                if diagnostics["candidate_count"] > 0:
                    belief_candidate_count_sum += diagnostics["candidate_count"]
                    belief_candidate_count_steps += 1

            # Predict step (prior): b' = f(b, a), then readout pre-correction o' = h(b').
            # Generated models may raise at any stage; keep evaluation running.
            try:
                belief_prior = _predict_belief_prior(model, belief, step.action)
            except Exception:
                belief_prior = belief
                predicted_obs = ""
            else:
                try:
                    predicted_obs = _readout_observation_from_belief(model, belief_prior, step.action)
                except Exception:
                    predicted_obs = ""

            if _is_non_error_prediction(predicted_obs):
                covered += 1
                placeholder_prediction = _is_placeholder_prediction(predicted_obs, traj_env_name)
                if placeholder_prediction:
                    placeholder_violations += 1
                else:
                    covered_non_placeholder += 1
                f1_sum += _token_f1(predicted_obs, step.next_observation)
                bleu4_sum += _safe_bleu4(predicted_obs, step.next_observation)
                if not placeholder_prediction:
                    non_placeholder_f1_sum += _token_f1(predicted_obs, step.next_observation)
                if predicted_obs == step.next_observation:
                    em_ok += 1
                    em_ok_all += 1
                try:
                    predicted_parse_attempts += 1
                    pred_parsed = model.parse_observation(predicted_obs)
                    if isinstance(pred_parsed, dict):
                        predicted_parse_ok += 1
                    actual_parsed = model.parse_observation(step.next_observation)
                    if isinstance(pred_parsed, dict) and isinstance(actual_parsed, dict):
                        state_covered += 1
                        comparison = compare_parsed_observations(pred_parsed, actual_parsed)
                        state_match_fraction_sum += comparison["field_match_fraction"]
                        if comparison["state_match"]:
                            state_match_ok += 1
                        if not placeholder_prediction:
                            state_covered_non_placeholder += 1
                            if comparison["state_match"]:
                                state_match_ok_non_placeholder += 1
                        for field_name in field_stats:
                            field_match = comparison["field_matches"].get(field_name)
                            if field_match is None:
                                continue
                            field_stats[field_name]["covered"] += 1
                            if field_match:
                                field_stats[field_name]["ok"] += 1
                except Exception:
                    pass
            else:
                # Not covered => cannot be exact match for "all transitions" metric
                pass

            # Correction step with the ground-truth next observation:
            # b = g(b', o_{t+1}); then optional post-correction readout o'' = h(b).
            try:
                belief = _correct_belief_with_observation(
                    model, belief_prior, step.next_observation
                )
            except Exception:
                # Keep prior belief if correction fails.
                belief = belief_prior
            try:
                corrected_obs = _readout_observation_from_belief(model, belief, step.action)
            except Exception:
                corrected_obs = ""
            if _is_non_error_prediction(corrected_obs):
                post_corrected_covered += 1
                post_corrected_f1_sum += _token_f1(corrected_obs, step.next_observation)

    mean_f1 = f1_sum / covered if covered > 0 else 0.0
    mean_bleu4 = bleu4_sum / covered if covered > 0 else 0.0
    mean_f1_post_correction = (
        post_corrected_f1_sum / post_corrected_covered if post_corrected_covered > 0 else 0.0
    )
    exact_match_rate = em_ok / covered if covered > 0 else 0.0
    exact_match_rate_all = em_ok_all / total if total > 0 else 0.0
    state_match_rate = state_match_ok / state_covered if state_covered > 0 else 0.0
    state_match_rate_all = state_match_ok / total if total > 0 else 0.0
    state_match_rate_non_placeholder = (
        state_match_ok_non_placeholder / state_covered_non_placeholder
        if state_covered_non_placeholder > 0 else 0.0
    )
    state_field_match_rate = (
        state_match_fraction_sum / state_covered if state_covered > 0 else 0.0
    )
    state_coverage = state_covered / total if total > 0 else 0.0
    coverage = covered / total if total > 0 else 0.0
    post_correction_coverage = post_corrected_covered / total if total > 0 else 0.0
    non_placeholder_coverage = covered_non_placeholder / total if total > 0 else 0.0
    mean_f1_non_placeholder = (
        non_placeholder_f1_sum / covered_non_placeholder if covered_non_placeholder > 0 else 0.0
    )
    placeholder_violation_rate = placeholder_violations / covered if covered > 0 else 0.0
    placeholder_violation_rate_all = placeholder_violations / total if total > 0 else 0.0
    predicted_parse_success_rate = (
        predicted_parse_ok / predicted_parse_attempts if predicted_parse_attempts > 0 else 0.0
    )
    predicted_parse_success_rate_all = predicted_parse_ok / total if total > 0 else 0.0
    sanctioned_token_f1 = mean_f1 * coverage * (1.0 - placeholder_violation_rate_all)
    field_metrics = {}
    for field_name, stats in field_stats.items():
        field_coverage = stats["covered"] / state_covered if state_covered > 0 else 0.0
        field_metrics[f"{field_name}_match_rate"] = round(
            stats["ok"] / stats["covered"], 4
        ) if stats["covered"] > 0 else 0.0
        field_metrics[f"{field_name}_match_rate_all"] = round(
            stats["ok"] / total, 4
        ) if total > 0 else 0.0
        field_metrics[f"{field_name}_coverage"] = round(field_coverage, 4)

    result = {
        "mean_token_f1_pre_correction": round(mean_f1, 4),
        "mean_token_f1_post_correction": round(mean_f1_post_correction, 4),
        "mean_token_f1": round(mean_f1, 4),
        "mean_bleu4": round(mean_bleu4, 4),
        "exact_match_rate": round(exact_match_rate, 4),
        "exact_match_rate_all": round(exact_match_rate_all, 4),
        "state_match_rate": round(state_match_rate, 4),
        "state_match_rate_all": round(state_match_rate_all, 4),
        "mean_state_field_match_rate": round(state_field_match_rate, 4),
        "state_coverage": round(state_coverage, 4),
        "coverage": round(coverage, 4),
        "post_correction_coverage": round(post_correction_coverage, 4),
        "non_placeholder_coverage": round(non_placeholder_coverage, 4),
        "mean_token_f1_non_placeholder": round(mean_f1_non_placeholder, 4),
        "state_match_rate_non_placeholder": round(state_match_rate_non_placeholder, 4),
        "placeholder_violation_rate": round(placeholder_violation_rate, 4),
        "placeholder_violation_rate_all": round(placeholder_violation_rate_all, 4),
        "predicted_parse_success_rate": round(predicted_parse_success_rate, 4),
        "predicted_parse_success_rate_all": round(predicted_parse_success_rate_all, 4),
        "sanctioned_token_f1": round(sanctioned_token_f1, 4),
        "mean_belief_candidate_count": round(
            belief_candidate_count_sum / belief_candidate_count_steps, 4
        ) if belief_candidate_count_steps > 0 else 0.0,
        "n_transitions": total,
    }
    result.update(field_metrics)
    return result


def compute_autoregressive_rollout_token_f1(
    model: Any,
    trajectories: List[Trajectory],
    *,
    max_horizon: int = 15,
    report_steps: Optional[List[int]] = None,
    progress_every: int = 0,
    progress_prefix: str = "",
    progress_logger: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """RQ2-style autoregressive rollout using ground-truth actions and predicted feedback."""
    env_name = _infer_env_name(model)
    if report_steps is None:
        report_steps = [1, 2, 3, 5, 8, 10, 15]
    step_set = {s for s in report_steps if s > 0}
    per_step_scores: Dict[int, List[float]] = {s: [] for s in sorted(step_set)}
    per_step_placeholder_counts: Dict[int, int] = {s: 0 for s in sorted(step_set)}
    episodes_used = 0
    eligible_episodes = sum(1 for traj in trajectories if traj.transitions)
    total_predictions = 0
    placeholder_predictions = 0

    def _emit(msg: str) -> None:
        if progress_logger is not None:
            progress_logger(msg)
            return
        now = datetime.now().strftime("%H:%M:%S")
        print(f"[{now}] {msg}", flush=True)

    if progress_every > 0:
        prefix = f"{progress_prefix} " if progress_prefix else ""
        _emit(
            f"{prefix}rollout start: eligible_episodes={eligible_episodes}, "
            f"max_horizon={max_horizon}, report_steps={sorted(step_set)}"
        )

    for traj in trajectories:
        if not traj.transitions:
            continue
        episodes_used += 1
        first_obs = traj.transitions[0].observation
        belief = _init_belief_from_first_observation(model, first_obs)

        for t, step in enumerate(traj.transitions[:max_horizon]):
            step_id = t + 1
            try:
                belief_prior = _predict_belief_prior(model, belief, step.action)
            except Exception:
                belief_prior = belief
                predicted_obs = ""
            else:
                try:
                    predicted_obs = _readout_observation_from_belief(model, belief_prior, step.action)
                except Exception:
                    predicted_obs = ""
            total_predictions += 1
            is_placeholder = _is_placeholder_prediction(predicted_obs, env_name)
            if is_placeholder:
                placeholder_predictions += 1

            if step_id in step_set:
                per_step_scores[step_id].append(_token_f1(predicted_obs, step.next_observation))
                if is_placeholder:
                    per_step_placeholder_counts[step_id] += 1

            feedback_obs = predicted_obs
            try:
                belief = _correct_belief_with_observation(model, belief_prior, feedback_obs)
            except Exception:
                belief = belief_prior

        if progress_every > 0 and episodes_used % progress_every == 0:
            prefix = f"{progress_prefix} " if progress_prefix else ""
            _emit(f"{prefix}rollout progress: episodes={episodes_used}/{eligible_episodes}")

    summary: Dict[str, Any] = {}
    for step in sorted(step_set):
        vals = per_step_scores.get(step, [])
        if vals:
            summary[str(step)] = {
                "mean_token_f1": round(mean(vals), 4),
                "std_token_f1": round(pstdev(vals), 4) if len(vals) > 1 else 0.0,
                "n": len(vals),
                "placeholder_violation_rate": round(
                    per_step_placeholder_counts.get(step, 0) / float(len(vals)), 4
                ),
            }
        else:
            summary[str(step)] = {
                "mean_token_f1": 0.0,
                "std_token_f1": 0.0,
                "n": 0,
                "placeholder_violation_rate": 0.0,
            }

    observed_step_means = [
        payload["mean_token_f1"]
        for payload in summary.values()
        if payload.get("n", 0) > 0
    ]
    observed_steps = [step for step in sorted(step_set) if summary[str(step)].get("n", 0) > 0]
    first_f1 = summary[str(observed_steps[0])]["mean_token_f1"] if observed_steps else 0.0
    last_f1 = summary[str(observed_steps[-1])]["mean_token_f1"] if observed_steps else 0.0

    result = {
        "max_horizon": max_horizon,
        "report_steps": sorted(step_set),
        "episodes_used": episodes_used,
        "token_f1_by_step": summary,
        "rollout_auc_token_f1": round(mean(observed_step_means), 4) if observed_step_means else 0.0,
        "rollout_degradation": round(first_f1 - last_f1, 4) if observed_steps else 0.0,
        "placeholder_violation_rate": round(
            placeholder_predictions / float(max(1, total_predictions)), 4
        ),
    }
    if progress_every > 0:
        prefix = f"{progress_prefix} " if progress_prefix else ""
        _emit(f"{prefix}rollout done: episodes={episodes_used}/{eligible_episodes}")
    return result


def build_skipped_train_metrics_split(
    train_trajs: List[Trajectory],
) -> Dict[str, Any]:
    """Placeholder train split when experience_fit / observation_fit are skipped."""
    n_traj = len(train_trajs)
    n_trans = sum(len(t.transitions) for t in train_trajs)
    return {
        "skipped": True,
        "reason": "skip_train_metrics",
        "experience_fit": {
            "skipped": True,
            "latent_belief_tracking_rate": 0.0,
            "n_trajectories": n_traj,
            "n_transitions": n_trans,
        },
        "observation_fit": {
            "skipped": True,
            "n_trajectories": n_traj,
            "n_transitions": n_trans,
            "coverage": 0.0,
        },
    }


def evaluate_world_model(
    model_code: str,
    train_trajs: List[Trajectory],
    eval_trajs: List[Trajectory],
    *,
    observation_fit_progress_every: int = 0,
    observation_fit_env_label: str = "",
    skip_train_metrics: bool = False,
) -> Dict:
    """Top-level evaluation: load model from generated code and run all metrics.

    Returns a nested dict:
    {
        "train": {"experience_fit": {...}, "observation_fit": {...}},
        "eval":  {"experience_fit": {...}, "observation_fit": {...}},
        "load_error": str | None,
    }
    """
    model, load_error = load_model_from_code(model_code)

    if load_error or model is None:
        stub = {
            "parse_success_rate": 0.0,
            "predict_success_rate": 0.0,
            "overall": 0.0,
            "latent_belief_tracking_rate": 0.0,
            "n_trajectories": 0,
            "n_transitions": 0,
        }
        obs_stub = {
            "mean_token_f1_pre_correction": 0.0,
            "mean_token_f1_post_correction": 0.0,
            "mean_token_f1": 0.0,
            "coverage": 0.0,
            "post_correction_coverage": 0.0,
            "state_match_rate": 0.0,
            "state_match_rate_all": 0.0,
            "mean_state_field_match_rate": 0.0,
            "state_coverage": 0.0,
            "room_match_rate": 0.0,
            "objects_match_rate": 0.0,
            "inventory_match_rate": 0.0,
            "n_transitions": 0,
        }
        return {
            "train": {"experience_fit": stub, "observation_fit": obs_stub},
            "eval": {"experience_fit": stub, "observation_fit": obs_stub},
            "load_error": load_error or "unknown load error",
        }

    result: Dict = {"load_error": None}
    if skip_train_metrics:
        result["train"] = build_skipped_train_metrics_split(train_trajs)
    else:
        prefix_tr = ""
        if observation_fit_env_label:
            prefix_tr = f"[{observation_fit_env_label}] symbolic train"
        result["train"] = {
            "experience_fit": compute_experience_fit(model, train_trajs),
            "observation_fit": compute_observation_fit(
                model,
                train_trajs,
                progress_every=observation_fit_progress_every,
                progress_prefix=prefix_tr,
            ),
        }
    prefix_ev = ""
    if observation_fit_env_label:
        prefix_ev = f"[{observation_fit_env_label}] symbolic eval"
    result["eval"] = {
        "experience_fit": compute_experience_fit(model, eval_trajs),
        "observation_fit": compute_observation_fit(
            model,
            eval_trajs,
            progress_every=observation_fit_progress_every,
            progress_prefix=prefix_ev,
        ),
    }
    return result
