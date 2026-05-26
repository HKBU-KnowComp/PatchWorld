import importlib.util
import os
import signal
import tempfile
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple

from patchworld.worldmodel_base import (
    belief_supports_latent_inference,
    coerce_maze_state,
    coerce_textcraft_state,
    coerce_webshop_state,
    coerce_wordle_state,
)
from patchworld.worldmodel_data import Trajectory
from patchworld.worldmodel_signals import belief_snapshot


@contextmanager
def _method_timeout(seconds: float):
    """Best-effort timeout for generated model method calls (Unix only)."""
    if seconds <= 0:
        yield
        return
    if not hasattr(signal, "setitimer"):
        yield
        return

    def _handler(signum, frame):  # pragma: no cover - runtime safety net
        raise TimeoutError(f"model method timed out after {seconds:.2f}s")

    old_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, old_handler)


def _call_with_timeout(
    fn,
    *args: Any,
    timeout_s: float,
):
    with _method_timeout(timeout_s):
        return fn(*args)


def _load_module_from_code(code: str):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        f.flush()
        path = f.name

    spec = importlib.util.spec_from_file_location("generated_worldmodel", path)
    module = importlib.util.module_from_spec(spec)
    if not spec or not spec.loader:
        raise RuntimeError("Failed to create module spec for generated world model.")
    spec.loader.exec_module(module)
    return module


def load_model_from_code(code: str) -> Tuple[Any, str]:
    """Dynamically load generated model code and return an instance."""
    try:
        module = _load_module_from_code(code)
    except Exception as e:
        return None, f"Code failed to load: {e}"

    from patchworld.worldmodel_base import BaseWorldModel

    candidates = []
    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if (
            isinstance(attr, type)
            and attr_name.endswith("WorldModel")
            and attr is not BaseWorldModel
            and issubclass(attr, BaseWorldModel)
        ):
            candidates.append((attr_name, attr))

    # Prefer the most-derived class (longest MRO = most specific subclass)
    candidates.sort(key=lambda x: len(x[1].__mro__), reverse=True)

    for attr_name, attr in candidates:
        try:
            return attr(), ""
        except Exception as e:
            return None, f"Failed to instantiate {attr_name}: {e}"

    # Fallback: accept duck-typed classes ending with WorldModel even when the
    # LLM forgets to subclass BaseWorldModel.
    required_methods = (
        "init_belief",
        "init_belief_from_observation",
        "predict_belief",
        "correct_belief",
        "readout_observation",
        "parse_observation",
        "extract_valid_action_forms",
    )
    duck_candidates = []
    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if not (isinstance(attr, type) and attr_name.endswith("WorldModel")):
            continue
        if all(callable(getattr(attr, method, None)) for method in required_methods):
            duck_candidates.append((attr_name, attr))

    for attr_name, attr in duck_candidates:
        try:
            return attr(), ""
        except Exception as e:
            return None, f"Failed to instantiate duck-typed {attr_name}: {e}"

    return None, "No loadable *WorldModel class found in generated code."


def validate_model(
    code: str,
    trajectories: List[Trajectory],
    max_trajectories: int = 20,
    show_progress: bool = False,
    progress_label: str = "",
    *,
    attach_step_text_to_first_n_errors: int = 0,
    method_timeout_s: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Replay trajectories through generated model, collect validation errors."""
    label = progress_label or "validate"

    model, load_error = load_model_from_code(code)
    if load_error:
        return [
            {
                "error_type": "load_error",
                "error": load_error,
                "action": "",
                "state": {},
                "expected": "",
                "got": "",
            }
        ]

    errors: List[Dict[str, Any]] = []
    timeout_s = (
        float(method_timeout_s)
        if method_timeout_s is not None
        else float(os.environ.get("PATCHWORLD_VALIDATOR_METHOD_TIMEOUT_S", "2.0"))
    )
    sample = trajectories[:max_trajectories]
    total = len(sample)
    if show_progress:
        print(f"[worldmodel_validator] {label}: start trajectories={total}")

    for traj_idx, traj in enumerate(sample, start=1):
        if show_progress and (traj_idx == 1 or traj_idx == total or traj_idx % 10 == 0):
            print(f"[worldmodel_validator] {label}: trajectory {traj_idx}/{total}")
        state: Dict[str, Any] = {}
        env_name = (getattr(traj, "env", "") or "").strip().lower()
        if traj.transitions:
            try:
                belief = model.init_belief_from_observation(traj.transitions[0].observation)
            except Exception:
                try:
                    belief = model.init_belief()
                except Exception:
                    belief = {}
        else:
            try:
                belief = model.init_belief()
            except Exception:
                belief = {}

        for step in traj.transitions:
            detail_remaining = [max(0, int(attach_step_text_to_first_n_errors))]
            predicted_obs_for_detail = ""

            def add_err(err: Dict[str, Any], *, pred_obs: Optional[str] = None) -> None:
                if detail_remaining[0] > 0:
                    err = dict(err)
                    err["observation"] = (step.observation or "")[:8000]
                    err["next_observation"] = (step.next_observation or "")[:8000]
                    po = (predicted_obs_for_detail if pred_obs is None else pred_obs) or ""
                    if po:
                        err["predicted_observation"] = po[:8000]
                    detail_remaining[0] -= 1
                errors.append(err)

            actual_next_parsed: Dict[str, Any] = {}
            try:
                parsed = _call_with_timeout(
                    model.parse_observation,
                    step.observation,
                    timeout_s=timeout_s,
                )
                if isinstance(parsed, dict):
                    state.update(parsed)
            except Exception as e:
                add_err(
                    {
                        "error_type": "parse_error",
                        "action": step.action,
                        "state": dict(state),
                        "expected": "successful parse",
                        "got": str(e),
                    }
                )
                continue

            try:
                belief = _call_with_timeout(
                    model.correct_belief,
                    belief,
                    step.observation,
                    timeout_s=timeout_s,
                )
            except Exception as e:
                add_err(
                    {
                        "error_type": "correct_belief_error",
                        "action": step.action,
                        "state": belief_snapshot(belief),
                        "expected": "correct_belief(belief, observation) without exception",
                        "got": str(e),
                    }
                )
            if env_name in {"wordle", "webshop", "alfworld", "babyai", "sciworld", "textcraft"}:
                if not belief_supports_latent_inference(belief, env_name):
                    add_err(
                        {
                            "error_type": "missing_latent_belief",
                            "action": step.action,
                            "state": belief_snapshot(belief),
                            "expected": f"belief with latent support for partial-observation env `{env_name}`",
                            "got": "belief contains only directly observed state/text",
                        }
                    )

            try:
                predicted_next_state = _call_with_timeout(
                    model.predict_belief,
                    belief,
                    step.action,
                    timeout_s=timeout_s,
                )
            except Exception as e:
                add_err(
                    {
                        "error_type": "predict_error",
                        "action": step.action,
                        "state": dict(state),
                        "expected": "successful prediction",
                        "got": str(e),
                    }
                )
                continue
            try:
                actual_next_parsed_candidate = _call_with_timeout(
                    model.parse_observation,
                    step.next_observation,
                    timeout_s=timeout_s,
                )
                if isinstance(actual_next_parsed_candidate, dict):
                    actual_next_parsed = actual_next_parsed_candidate
            except Exception:
                actual_next_parsed = {}
            current_maze = coerce_maze_state(state)
            next_maze = coerce_maze_state(actual_next_parsed)
            if (current_maze is not None or next_maze is not None) and isinstance(predicted_next_state, dict):
                from patchworld.worldmodel_evaluator import compare_parsed_observations

                transition_comparison = compare_parsed_observations(predicted_next_state, actual_next_parsed)
                if not transition_comparison["state_match"]:
                    add_err(
                        {
                            "error_type": "maze_state_transition_mismatch",
                            "action": step.action,
                            "state": belief_snapshot(belief),
                            "expected": str(transition_comparison["actual_fields"])[:300],
                            "got": str(transition_comparison["predicted_fields"])[:300],
                        }
                    )

            predicted_obs = ""
            try:
                predicted_obs = (
                    _call_with_timeout(
                        model.readout_observation,
                        predicted_next_state,
                        step.action,
                        timeout_s=timeout_s,
                    )
                    or ""
                )
            except Exception as e:
                add_err(
                    {
                        "error_type": "predict_observation_error",
                        "action": step.action,
                        "state": belief_snapshot(belief),
                        "expected": "readout_observation(predicted_belief, action) without exception",
                        "got": str(e),
                    }
                )
            predicted_obs_for_detail = predicted_obs
            if not str(predicted_obs or "").strip():
                add_err(
                    {
                        "error_type": "empty_readout_observation",
                        "action": step.action,
                        "state": belief_snapshot(belief),
                        "expected": "non-empty predicted next-observation text from readout_observation",
                        "got": "empty string",
                    }
                )
            if predicted_obs:
                try:
                    predicted_parsed = _call_with_timeout(
                        model.parse_observation,
                        predicted_obs,
                        timeout_s=timeout_s,
                    )
                    actual_parsed = actual_next_parsed or _call_with_timeout(
                        model.parse_observation,
                        step.next_observation,
                        timeout_s=timeout_s,
                    )
                    if isinstance(predicted_parsed, dict) and isinstance(actual_parsed, dict):
                        from patchworld.worldmodel_evaluator import compare_parsed_observations

                        comparison = compare_parsed_observations(predicted_parsed, actual_parsed)
                        if not comparison["state_match"]:
                            add_err(
                                {
                                    "error_type": "render_roundtrip_mismatch",
                                    "action": step.action,
                                    "state": belief_snapshot(belief),
                                    "expected": str(comparison["actual_fields"])[:300],
                                    "got": str(comparison["predicted_fields"])[:300],
                                }
                            )
                        predicted_maze = coerce_maze_state(predicted_parsed)
                        actual_maze = coerce_maze_state(actual_parsed)
                        if (
                            actual_maze is not None
                            and actual_maze.status != "success"
                            and actual_maze.local_walls
                            and predicted_maze is not None
                            and not predicted_maze.local_walls
                        ):
                            add_err(
                                {
                                    "error_type": "maze_render_missing_walls",
                                    "action": step.action,
                                    "state": belief_snapshot(belief),
                                    "expected": str(actual_maze.local_walls),
                                    "got": str(predicted_maze.local_walls),
                                }
                            )
                        predicted_wordle = coerce_wordle_state(predicted_parsed)
                        actual_wordle = coerce_wordle_state(actual_parsed)
                        if (
                            actual_wordle is not None
                            and actual_wordle.current_feedback
                            and (
                                predicted_wordle is None
                                or predicted_wordle.current_feedback != actual_wordle.current_feedback
                            )
                        ):
                            add_err(
                                {
                                    "error_type": "wordle_feedback_mismatch",
                                    "action": step.action,
                                    "state": belief_snapshot(belief),
                                    "expected": str(actual_wordle.current_feedback),
                                    "got": str(
                                        predicted_wordle.current_feedback if predicted_wordle is not None else ()
                                    ),
                                }
                            )
                        predicted_textcraft = coerce_textcraft_state(predicted_parsed)
                        actual_textcraft = coerce_textcraft_state(actual_parsed)
                        if (
                            actual_textcraft is not None
                            and actual_textcraft.inventory
                            and (
                                predicted_textcraft is None
                                or predicted_textcraft.inventory != actual_textcraft.inventory
                            )
                        ):
                            add_err(
                                {
                                    "error_type": "textcraft_inventory_mismatch",
                                    "action": step.action,
                                    "state": belief_snapshot(belief),
                                    "expected": str(actual_textcraft.inventory),
                                    "got": str(
                                        predicted_textcraft.inventory if predicted_textcraft is not None else ()
                                    ),
                                }
                            )
                        predicted_webshop = coerce_webshop_state(predicted_parsed)
                        actual_webshop = coerce_webshop_state(actual_parsed)
                        if (
                            actual_webshop is not None
                            and actual_webshop.page_type
                            and (
                                predicted_webshop is None
                                or predicted_webshop.page_type != actual_webshop.page_type
                            )
                        ):
                            add_err(
                                {
                                    "error_type": "webshop_page_state_mismatch",
                                    "action": step.action,
                                    "state": belief_snapshot(belief),
                                    "expected": actual_webshop.page_type,
                                    "got": predicted_webshop.page_type if predicted_webshop is not None else "",
                                }
                            )
                except Exception as e:
                    add_err(
                        {
                            "error_type": "render_roundtrip_parse_error",
                            "action": step.action,
                            "state": belief_snapshot(belief),
                            "expected": "parse_observation(readout_observation(...)) without exception",
                            "got": str(e),
                        }
                    )

            if isinstance(actual_next_parsed, dict):
                state.update(actual_next_parsed)
            try:
                belief_prior = _call_with_timeout(
                    model.predict_belief,
                    belief,
                    step.action,
                    timeout_s=timeout_s,
                )
                belief = _call_with_timeout(
                    model.correct_belief,
                    belief_prior,
                    step.next_observation,
                    timeout_s=timeout_s,
                )
            except Exception as e:
                add_err(
                    {
                        "error_type": "next_correct_belief_error",
                        "action": step.action,
                        "state": belief_snapshot(belief),
                        "expected": "predict_belief(belief, action) + correct_belief(prior, next_observation)",
                        "got": str(e),
                    }
                )

    if show_progress:
        print(f"[worldmodel_validator] {label}: done errors={len(errors)}")
    return errors

