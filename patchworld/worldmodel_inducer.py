"""World-model induction via LLM code generation and replay feedback."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json
import uuid

from patchworld.inducer_constants import SPATIAL_ENVS
from patchworld.inducer_llm_client import LLMClient
from patchworld.inducer_prompts import (
    API_CODING_PROMPT,
    EPISTEMIC_CODING_PROMPT,
    ENV_PROMPT_GUIDANCE,
    REFINE_PROMPT,
    SPATIAL_CODING_PROMPT,
)
from patchworld.inducer_transition_sampling import (
    _action_type,
    _outcome_type,
    _sciworld_task_family,
    _task_text_from_observation,
    select_contrastive_examples,
    select_random_examples,
)
from patchworld.worldmodel_data import Trajectory, Transition
from patchworld.worldmodel_validator import validate_model

NAIVE_INDUCTION = "naive_induction"
TRACEPATCH_INDUCTION = "tracepatch"
TRACEPATCH_LOAD_ERROR_SCORE = 1_000_000.0

__all__ = ["Inducer", "LLMClient", "NAIVE_INDUCTION", "TRACEPATCH_INDUCTION"]


class Inducer:
    """World-model inducer with naive and agentic repair strategies.

    ``naive_induction`` is the original one-candidate loop: sample examples,
    generate code, validate, and blindly replace the program with each LLM
    refinement.

    ``tracepatch`` is an agentic counterexample-guided patch search: run the
    formal replay tests, diagnose and prioritize failures, generate candidate
    patches, validate every patch, and accept only improving candidates.
    """

    def __init__(
        self,
        llm: LLMClient,
        max_refine_rounds: int = 3,
        holdout_fraction: float = 0.0,
        contrastive_max_per_pattern: int = 5,
        contrastive_max_transitions: int = 80,
        disable_stage1_rules: bool = True,
        random_transition_sampling: bool = False,
        verbose: bool = False,
        debug: bool = False,
        show_progress: bool = False,
        sgd_epochs: int = 1,
        sgd_batch_size: int = 20,
        sgd_early_stop_patience: int = 0,
        sgd_min_delta: float = 0.0,
        metric_max_trajectories: int = 20,
        metric_max_transitions: int = 200,
        snapshot_dir: str | Path | None = None,
        induction_track: str = "simple_state",
        selection_profile: str = "default",
        induction_strategy: str = TRACEPATCH_INDUCTION,
        patch_candidates_per_round: int = 2,
        patch_beam_size: int = 1,
        patch_no_improve_patience: int = 0,
        initial_load_error_retries: int = 5,
        disable_validation_gate: bool = False,
        disable_interface_constraints: bool = False,
        **legacy_kwargs: Any,
    ) -> None:
        self.llm = llm
        self.max_refine_rounds = max(0, int(max_refine_rounds))
        self.contrastive_max_per_pattern = max(1, int(contrastive_max_per_pattern))
        self.contrastive_max_transitions = max(1, int(contrastive_max_transitions))
        self.verbose = verbose
        self.debug = debug
        self.show_progress = show_progress
        self.snapshot_dir = Path(snapshot_dir) if snapshot_dir else None
        self.last_induce_stats: Dict[str, Any] = {}
        self._snapshot_counter = 0
        self._version_records: List[Dict[str, Any]] = []

        # Kept only for compatibility with existing scripts/configs.
        self.holdout_fraction = holdout_fraction
        self.disable_stage1_rules = disable_stage1_rules
        self.random_transition_sampling = random_transition_sampling
        self.sgd_epochs = sgd_epochs
        self.sgd_batch_size = sgd_batch_size
        self.sgd_early_stop_patience = sgd_early_stop_patience
        self.sgd_min_delta = sgd_min_delta
        self.metric_max_trajectories = metric_max_trajectories
        self.metric_max_transitions = metric_max_transitions
        self.induction_track = self._normalize_induction_track(induction_track)
        self.selection_profile = self._normalize_selection_profile(selection_profile)
        self.induction_strategy = self._normalize_induction_strategy(induction_strategy)
        self.patch_candidates_per_round = max(1, int(patch_candidates_per_round))
        self.patch_beam_size = max(1, int(patch_beam_size))
        self.patch_no_improve_patience = max(0, int(patch_no_improve_patience))
        self.initial_load_error_retries = max(0, int(initial_load_error_retries))
        self.disable_validation_gate = bool(disable_validation_gate)
        self.disable_interface_constraints = bool(disable_interface_constraints)
        self.legacy_kwargs = dict(legacy_kwargs)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def induce(
        self,
        env_name: str,
        trajectories: List[Trajectory],
        validation_trajectories: Optional[List[Trajectory]] = None,
    ) -> str:
        """Generate a world-model implementation with the configured strategy."""
        if self.induction_strategy == NAIVE_INDUCTION:
            return self.naive_induce(env_name, trajectories, validation_trajectories)
        return self.tracepatch_induce(env_name, trajectories, validation_trajectories)

    def naive_induce(
        self,
        env_name: str,
        trajectories: List[Trajectory],
        validation_trajectories: Optional[List[Trajectory]] = None,
    ) -> str:
        """Run the original naive induction loop.

        This path is intentionally simple: every repair emitted by the LLM
        replaces the current program before the next validation pass.
        """
        env = (env_name or "").strip().lower()
        is_spatial = env in SPATIAL_ENVS
        class_name = f"{env.capitalize()}WorldModel" if env else "WorldModel"
        validation_trajs = validation_trajectories or trajectories

        self._snapshot_counter = 0
        self._version_records = []
        self._progress(
            f"starting naive induction env={env_name} trajectories={len(trajectories)}"
        )

        transitions = self._select_induction_transitions(trajectories)
        transition_text = self._format_transitions(transitions)
        rules = (
            "No separate rule extraction is used. Infer the transition dynamics "
            "directly from the contrastive examples."
        )
        prompt = self._build_coding_prompt(
            env_name,
            is_spatial,
            class_name,
            rules,
            transition_text,
        )

        code = self._extract_code(self.llm.generate(prompt))
        self._save_model_snapshot(
            env_name=env_name,
            phase="initial",
            code=code,
            metadata={"selected_transitions": len(transitions)},
        )

        errors = self._validate(code, validation_trajs, label="initial")
        initial_error_count = len(errors)
        round_history: List[Dict[str, Any]] = []

        for round_idx in range(1, self.max_refine_rounds + 1):
            if not errors:
                break

            self._progress(
                f"refine round {round_idx}/{self.max_refine_rounds}: errors={len(errors)}"
            )
            error_text = self._format_errors(errors[:12])
            code = self._extract_code(
                self.llm.generate(
                    REFINE_PROMPT.format(
                        code=code,
                        errors=error_text,
                        error_summary=self._summarize_errors(errors),
                        rules_guidance=rules,
                    )
                )
            )
            self._save_model_snapshot(
                env_name=env_name,
                phase=f"round_{round_idx:02d}",
                code=code,
                metadata={"input_error_count": len(errors)},
            )
            errors = self._validate(code, validation_trajs, label=f"round_{round_idx:02d}")
            round_history.append(
                {
                    "round": round_idx,
                    "error_count": len(errors),
                    "converged": not errors,
                }
            )

        final_errors = errors
        converged = not final_errors
        self._save_model_snapshot(
            env_name=env_name,
            phase="final",
            code=code,
            metadata={"final_error_count": len(final_errors), "converged": converged},
        )
        self.last_induce_stats = {
            "total_trajectories": len(trajectories),
            "validation_trajectories": len(validation_trajs),
            "selected_transitions": len(transitions),
            "max_refine_rounds": self.max_refine_rounds,
            "initial_error_count": initial_error_count,
            "final_error_count": len(final_errors),
            "rounds_executed": len(round_history),
            "converged": converged,
            "history": round_history,
            "snapshot_dir": str(self.snapshot_dir) if self.snapshot_dir else None,
            "saved_versions": list(self._version_records),
            "induction_method": NAIVE_INDUCTION,
            "induction_track": self.induction_track,
            "selection_profile": self.selection_profile,
            "simplified_inducer": True,
        }
        self._progress(
            f"naive induction finished converged={converged} final_errors={len(final_errors)}"
        )
        return code

    def tracepatch_induce(
        self,
        env_name: str,
        trajectories: List[Trajectory],
        validation_trajectories: Optional[List[Trajectory]] = None,
    ) -> str:
        """Run TracePatch, an agentic counterexample-guided repair loop."""
        env = (env_name or "").strip().lower()
        is_spatial = env in SPATIAL_ENVS
        class_name = f"{env.capitalize()}WorldModel" if env else "WorldModel"
        validation_trajs = validation_trajectories or trajectories

        self._snapshot_counter = 0
        self._version_records = []
        self._progress(
            f"starting TracePatch induction env={env_name} trajectories={len(trajectories)}"
        )

        transitions = self._select_induction_transitions(trajectories)
        transition_text = self._format_transitions(transitions)
        rules = (
            "No separate rule extraction is used. Infer the transition dynamics "
            "directly from the contrastive examples."
        )
        prompt = self._build_coding_prompt(
            env_name,
            is_spatial,
            class_name,
            rules,
            transition_text,
        )

        code = self._extract_code(self.llm.generate(prompt))
        self._save_model_snapshot(
            env_name=env_name,
            phase="initial",
            code=code,
            metadata={"selected_transitions": len(transitions), "accepted": True},
        )

        errors = self._run_formal_tests(code, validation_trajs, label="initial_full")
        current_score = self._score_formal_test_result(errors)
        initial_error_count = len(errors)
        if self._has_load_error(errors):
            best_code = code
            best_errors = errors
            best_score = current_score
            retry_count = max(1, self.initial_load_error_retries)
            self._progress(
                f"initial model has load error; trying {retry_count} targeted load-error retries"
            )
            for retry_idx in range(1, retry_count + 1):
                retry_prompt = self._build_load_error_retry_prompt(
                    env_name=env_name,
                    original_prompt=prompt,
                    bad_code=best_code,
                    errors=best_errors,
                    retry_idx=retry_idx,
                )
                retry_code = self._extract_code(self.llm.generate(retry_prompt))
                retry_errors = self._run_formal_tests(
                    retry_code,
                    validation_trajs,
                    label=f"initial_retry_{retry_idx:02d}",
                )
                retry_score = self._score_formal_test_result(retry_errors)
                self._save_model_snapshot(
                    env_name=env_name,
                    phase=f"initial_retry_{retry_idx:02d}",
                    code=retry_code,
                    metadata={
                        "error_count": len(retry_errors),
                        "score": retry_score,
                        "accepted": False,
                    },
                )
                if retry_score < best_score:
                    best_code = retry_code
                    best_errors = retry_errors
                    best_score = retry_score
            code = best_code
            errors = best_errors
            current_score = best_score
        round_history: List[Dict[str, Any]] = []
        no_improve_rounds = 0

        for round_idx in range(1, self.max_refine_rounds + 1):
            if not errors:
                break

            diagnosis = self._diagnose_failures(errors)
            prioritized_errors = self._prioritize_failures(errors)
            self._progress(
                "TracePatch round "
                f"{round_idx}/{self.max_refine_rounds}: "
                f"errors={len(errors)} score={current_score:.3f}"
            )

            candidates: List[Dict[str, Any]] = []
            for candidate_idx in range(1, self.patch_candidates_per_round + 1):
                patch_prompt = self._build_tracepatch_prompt(
                    env_name=env_name,
                    code=code,
                    errors=prioritized_errors,
                    diagnosis=diagnosis,
                    rules=rules,
                    round_idx=round_idx,
                    candidate_idx=candidate_idx,
                )
                candidate_code = self._extract_code(self.llm.generate(patch_prompt))
                candidate_errors = self._run_formal_tests(
                    candidate_code,
                    validation_trajs,
                    label=f"round_{round_idx:02d}_candidate_{candidate_idx:02d}",
                )
                candidate_score = self._score_formal_test_result(candidate_errors)
                candidate = {
                    "round": round_idx,
                    "candidate": candidate_idx,
                    "code": candidate_code,
                    "errors": candidate_errors,
                    "error_count": len(candidate_errors),
                    "score": candidate_score,
                }
                candidates.append(candidate)

            candidates.sort(key=lambda item: (item["score"], item["error_count"]))
            accepted = False
            best = candidates[0] if candidates else None
            for candidate in candidates[: self.patch_beam_size]:
                improves, reason = self._patch_improves(
                    current_score=current_score,
                    current_errors=errors,
                    candidate_score=candidate["score"],
                    candidate_errors=candidate["errors"],
                    load_error_acceptance_cap=max(1000, len(validation_trajs) * 10),
                )
                if self.disable_validation_gate and not accepted:
                    improves = True
                    reason = f"validation_gate_disabled:{reason}"
                candidate["accepted"] = improves
                candidate["accept_reason"] = reason
                self._save_model_snapshot(
                    env_name=env_name,
                    phase=(
                        f"round_{round_idx:02d}_candidate_"
                        f"{candidate['candidate']:02d}"
                    ),
                    code=candidate["code"],
                    metadata={
                        "input_error_count": len(errors),
                        "error_count": candidate["error_count"],
                        "score": candidate["score"],
                        "accepted": improves,
                        "accept_reason": reason,
                        "diagnosis": diagnosis,
                    },
                )
                if improves and not accepted:
                    best = candidate
                    accepted = True

            if accepted and best is not None:
                code = best["code"]
                errors = best["errors"]
                current_score = best["score"]
                no_improve_rounds = 0
            else:
                no_improve_rounds += 1
                should_continue = no_improve_rounds <= self.patch_no_improve_patience
                round_history.append(
                    {
                        "round": round_idx,
                        "accepted": False,
                        "reason": (
                            "no_improving_patch_retry"
                            if should_continue
                            else "no_improving_patch"
                        ),
                        "error_count": len(errors),
                        "score": current_score,
                        "no_improve_rounds": no_improve_rounds,
                        "diagnosis": diagnosis,
                    }
                )
                if should_continue:
                    self._progress(
                        "TracePatch continuing after non-improving batch "
                        f"({no_improve_rounds}/{self.patch_no_improve_patience})"
                    )
                    continue
                break

            round_history.append(
                {
                    "round": round_idx,
                    "accepted": True,
                    "accepted_candidate": best["candidate"],
                    "error_count": len(errors),
                    "score": current_score,
                    "converged": not errors,
                    "diagnosis": diagnosis,
                }
            )

        final_errors = errors
        converged = not final_errors
        self._save_model_snapshot(
            env_name=env_name,
            phase="final",
            code=code,
            metadata={
                "final_error_count": len(final_errors),
                "final_score": current_score,
                "converged": converged,
                "accepted": True,
            },
        )
        self.last_induce_stats = {
            "total_trajectories": len(trajectories),
            "validation_trajectories": len(validation_trajs),
            "selected_transitions": len(transitions),
            "max_refine_rounds": self.max_refine_rounds,
            "patch_candidates_per_round": self.patch_candidates_per_round,
            "patch_beam_size": self.patch_beam_size,
            "patch_no_improve_patience": self.patch_no_improve_patience,
            "initial_error_count": initial_error_count,
            "final_error_count": len(final_errors),
            "rounds_executed": len(round_history),
            "converged": converged,
            "history": round_history,
            "snapshot_dir": str(self.snapshot_dir) if self.snapshot_dir else None,
            "saved_versions": list(self._version_records),
            "induction_method": TRACEPATCH_INDUCTION,
            "induction_track": self.induction_track,
            "selection_profile": self.selection_profile,
            "random_transition_sampling": self.random_transition_sampling,
            "disable_validation_gate": self.disable_validation_gate,
            "disable_interface_constraints": self.disable_interface_constraints,
            "agentic_fix_loop": True,
        }
        self._progress(
            f"TracePatch induction finished converged={converged} final_errors={len(final_errors)}"
        )
        return code

    def refine(
        self,
        code: str,
        new_trajectories: List[Trajectory],
        max_rounds: int = 2,
    ) -> str:
        """Incrementally refine existing code using replay errors only."""
        errors = self._validate(code, new_trajectories, label="refine_initial")
        for round_idx in range(1, max(0, int(max_rounds)) + 1):
            if not errors:
                break
            code = self._extract_code(
                self.llm.generate(
                    REFINE_PROMPT.format(
                        code=code,
                        errors=self._format_errors(errors[:12]),
                        error_summary=self._summarize_errors(errors),
                        rules_guidance="No separate rules are available; preserve working behavior.",
                    )
                )
            )
            errors = self._validate(code, new_trajectories, label=f"refine_{round_idx:02d}")
        return code

    # ------------------------------------------------------------------
    # Small helpers
    # ------------------------------------------------------------------

    def _normalize_induction_strategy(self, strategy: str) -> str:
        normalized = (strategy or TRACEPATCH_INDUCTION).strip().lower().replace("-", "_")
        aliases = {
            "minimal": NAIVE_INDUCTION,
            "naive": NAIVE_INDUCTION,
            "naive_induction": NAIVE_INDUCTION,
            "trace_patch": TRACEPATCH_INDUCTION,
            "tracepatch": TRACEPATCH_INDUCTION,
            "agentic": TRACEPATCH_INDUCTION,
            "agentic_fix_loop": TRACEPATCH_INDUCTION,
        }
        return aliases.get(normalized, TRACEPATCH_INDUCTION)

    def _normalize_induction_track(self, track: str) -> str:
        normalized = (track or "simple_state").strip().lower().replace("-", "_")
        aliases = {
            "symbolic": "simple_state",
            "simple": "simple_state",
            "simple_state": "simple_state",
            "epistemic": "epistemic_state",
            "epistemic_state": "epistemic_state",
            "neurosymbolic": "epistemic_state",
            "hybrid": "epistemic_state",
        }
        return aliases.get(normalized, normalized or "simple_state")

    def _normalize_selection_profile(self, profile: str) -> str:
        normalized = (profile or "default").strip().lower().replace("-", "_")
        return normalized or "default"

    def _build_coding_prompt(
        self,
        env_name: str,
        is_spatial: bool,
        class_name: str,
        rules: str,
        transition_text: str,
    ) -> str:
        if self.disable_interface_constraints:
            return self._build_loose_interface_prompt(
                env_name=env_name,
                class_name=class_name,
                rules=rules,
                transition_text=transition_text,
            )
        if self.induction_track == "epistemic_state":
            prompt_template = EPISTEMIC_CODING_PROMPT
        else:
            prompt_template = SPATIAL_CODING_PROMPT if is_spatial else API_CODING_PROMPT
        env_guidance = self._environment_guidance(env_name)
        if env_guidance:
            rules = (
                f"{rules}\n\n"
                f"Environment-specific requirements for `{env_name}`:\n{env_guidance}"
            )
        track_guidance = self._track_guidance()
        if track_guidance:
            rules = (
                f"{rules}\n\n"
                f"Induction-track requirements for `{self.induction_track}`:\n{track_guidance}"
            )
        return prompt_template.format(
            env_name=env_name,
            rules=rules,
            transition_examples=transition_text,
            class_name=class_name,
        )

    def _select_induction_transitions(self, trajectories: List[Trajectory]) -> List[Transition]:
        if self.random_transition_sampling:
            return select_random_examples(
                trajectories,
                max_transitions=self.contrastive_max_transitions,
            )
        return select_contrastive_examples(
            trajectories,
            max_transitions=self.contrastive_max_transitions,
            max_per_pattern=self.contrastive_max_per_pattern,
        )

    def _build_loose_interface_prompt(
        self,
        *,
        env_name: str,
        class_name: str,
        rules: str,
        transition_text: str,
    ) -> str:
        return f"""You are implementing a Python world model for the environment: {env_name}.

## Transition hints
{rules}

## Example transitions
{transition_text}

Write a complete Python module containing class {class_name}(BaseWorldModel).
The class should predict the next observation for new held-out trajectories.

For this ablation, do not use the full TracePatch interface guidance. You may
choose any internal representation that seems useful, but the module must still
load under the benchmark and implement the BaseWorldModel methods:

```python
from patchworld.worldmodel_base import BaseWorldModel

class {class_name}(BaseWorldModel):
    def parse_observation(self, obs_text: str) -> dict:
        ...

    def init_belief(self):
        ...

    def correct_belief(self, belief_prior, obs_text: str):
        ...

    def predict_belief(self, belief, action: str):
        ...

    def readout_observation(self, belief, action: str = "") -> str:
        ...

    def extract_valid_action_forms(self) -> dict[str, list[str]]:
        ...
```

Keep the code general: do not hardcode task-instance constants from the examples.
Output ONLY the Python code, nothing else.
"""

    def _validate(
        self,
        code: str,
        trajectories: List[Trajectory],
        *,
        label: str,
    ) -> List[Dict[str, Any]]:
        if not trajectories:
            return []
        return validate_model(
            code,
            trajectories,
            max_trajectories=min(20, len(trajectories)),
            show_progress=self.show_progress,
            progress_label=label,
            attach_step_text_to_first_n_errors=3,
        )

    def _run_formal_tests(
        self,
        code: str,
        trajectories: List[Trajectory],
        *,
        label: str,
    ) -> List[Dict[str, Any]]:
        """Run the configured full replay-test suite for TracePatch."""
        if not trajectories:
            return []
        return validate_model(
            code,
            trajectories,
            max_trajectories=len(trajectories),
            show_progress=self.show_progress,
            progress_label=label,
            attach_step_text_to_first_n_errors=6,
        )

    def _build_tracepatch_prompt(
        self,
        *,
        env_name: str,
        code: str,
        errors: List[Dict[str, Any]],
        diagnosis: Dict[str, Any],
        rules: str,
        round_idx: int,
        candidate_idx: int,
    ) -> str:
        env_guidance = self._environment_guidance(env_name)
        track_guidance = self._track_guidance()
        diagnosis_guidance = self._diagnosis_guidance(env_name, diagnosis)
        has_load_error = any(
            str(cluster.get("error_type", "") or "") == "load_error"
            for cluster in (diagnosis.get("clusters", []) if isinstance(diagnosis, dict) else [])
            if isinstance(cluster, dict)
        )
        cache_bust_line = (
            f"Load-error cache_bust_nonce: {uuid.uuid4().hex}\n"
            if has_load_error
            else ""
        )
        guidance = (
            f"{rules}\n\n"
            f"Environment-specific repair requirements for `{env_name}`:\n"
            f"{env_guidance or '(none)'}\n\n"
            "Diagnosis-specific repair requirements:\n"
            f"{diagnosis_guidance or '(none)'}\n\n"
            f"{cache_bust_line}"
            f"Induction-track repair requirements for `{self.induction_track}`:\n"
            f"{track_guidance or '(none)'}\n\n"
            "TracePatch repair protocol:\n"
            "1. Fix the prioritized failing replay tests before adding new behavior.\n"
            "2. Preserve behavior that is not implicated by the diagnosis.\n"
            "3. Prefer small, general patches over rewrites.\n"
            "4. Treat raw OBS / EXPECTED_NEXT / PREDICTED_OBS below as the primary "
            "evidence when normalized fields are empty or ambiguous.\n"
            "5. Keep parse_observation, predict_belief, correct_belief, and "
            "readout_observation on one canonical schema.\n"
            "6. This is candidate patch "
            f"{candidate_idx} for repair round {round_idx}; make an independently useful fix."
        )
        return REFINE_PROMPT.format(
            code=code,
            errors=self._format_errors(errors[:16]),
            error_summary=self._format_diagnosis(diagnosis),
            rules_guidance=guidance,
        )

    def _build_load_error_retry_prompt(
        self,
        *,
        env_name: str,
        original_prompt: str,
        bad_code: str,
        errors: List[Dict[str, Any]],
        retry_idx: int,
    ) -> str:
        env_guidance = self._environment_guidance(env_name)
        error_text = self._format_errors(errors[:4]) or self._summarize_errors(errors)
        # Load-error retries are intentionally cache-distinct. If a syntax-broken
        # completion is cached, reusing the exact repair prompt can deterministically
        # replay the same broken candidate across benchmark reruns.
        cache_bust_nonce = uuid.uuid4().hex
        return (
            f"{original_prompt}\n\n"
            f"# Targeted load-error retry {retry_idx}\n"
            f"# cache_bust_nonce: {cache_bust_nonce}\n"
            "The previous generated Python module failed to load. Return a complete, "
            "loadable Python implementation only.\n\n"
            "Load failure details:\n"
            f"{error_text}\n\n"
            "Previous failing code:\n"
            "```python\n"
            f"{bad_code}\n"
            "```\n\n"
            "Repair requirements:\n"
            "- Fix syntax/import/class/instantiation errors before changing behavior.\n"
            "- Every if/elif/else/for/while/try/except/function/class body must contain a real "
            "statement; use `pass` when the intended behavior is no-op. Comments alone are invalid.\n"
            "- Keep all required methods implemented: init_belief, init_belief_from_observation, "
            "parse_observation, predict_belief, correct_belief, readout_observation, "
            "extract_valid_action_forms.\n"
            "- `readout_observation` must return non-empty next-observation text after "
            "`predict_belief`; do not return only a cleared raw observation cache.\n"
            f"- Environment guidance for `{env_name}`: {env_guidance or '(none)'}\n"
        )

    def _diagnosis_guidance(self, env_name: str, diagnosis: Dict[str, Any]) -> str:
        """Translate recurring validator buckets into concrete repair directives."""
        env = (env_name or "").strip().lower()
        clusters = diagnosis.get("clusters", []) if isinstance(diagnosis, dict) else []
        error_types = {
            str(cluster.get("error_type", "") or "")
            for cluster in clusters
            if isinstance(cluster, dict)
        }
        guidance: List[str] = []

        if "missing_latent_belief" in error_types:
            guidance.append(
                "- The validator is explicitly rejecting a direct-observation cache. "
                "Add recognized latent-belief fields such as `latent_variables`, "
                "`facts`, `hypotheses`, `frontier`, or `hidden_state`, and keep them "
                "populated across init/correct/predict instead of adding unused keys."
            )
            if env == "textcraft":
                guidance.append(
                    "- For TextCraft, represent hidden availability/recipe knowledge with "
                    "`latent_variables` containing `obtainable_items`, "
                    "`known_unavailable_items`, and `recipe_graph`; expose canonical "
                    "`crafting_recipes`, `goal_item`, and `last_action_outcome` fields "
                    "so state extraction can evaluate inventory, recipes, and outcomes."
                )

        if "maze_state_transition_mismatch" in error_types or "maze_render_missing_walls" in error_types:
            guidance.append(
                "- Maze transition failures mean coordinates/status/walls are inconsistent. "
                "Use canonical wall directions `up`, `down`, `left`, `right`; parse only "
                "the final environment-state block; and only set terminal success from an "
                "exact terminal observation, not from instructional text containing the word Success."
            )
            guidance.append(
                "- Maintain latent map memory such as `wall_map[(agent_x, agent_y)]` and "
                "`visited_cells`. `local_walls` is the current cell's observed walls, while "
                "`wall_map` preserves walls learned for earlier cells and supports blocked "
                "movement/action-conditioned prediction."
            )

        if "render_roundtrip_mismatch" in error_types:
            guidance.append(
                "- Round-trip failures mean `readout_observation` emits text that "
                "`parse_observation` cannot recover. Patch parser and renderer together, "
                "then verify that parsing the rendered text recovers the same canonical fields."
            )
            if env == "maze":
                guidance.append(
                    "- For Maze, never render `Success` unless status is terminal. Non-terminal "
                    "rendering must include goal coordinates, current coordinates, and every "
                    "known current-cell wall in the observed template."
                )

        if "empty_readout_observation" in error_types:
            guidance.append(
                "- `readout_observation` returned empty text after `predict_belief`. "
                "Do not rely only on a raw observation cache that is cleared during prediction; "
                "synthesize a non-empty next-observation string from belief state and action."
            )
            if env == "textcraft":
                guidance.append(
                    "- For TextCraft, render inventory actions as an `Inventory:` block; "
                    "render successful get/craft actions as `Got <count> <item>` or "
                    "`Crafted <count> minecraft:<item_slug>`; and render failed crafts with "
                    "the observed `Could not find ...` templates."
                )

        if "textcraft_inventory_mismatch" in error_types:
            guidance.append(
                "- TextCraft inventory mismatches usually come from parsing or naming drift. "
                "Parse inventory entries as `[item] (count)`, normalize item names consistently, "
                "and make `get`/`craft` consume and add counts only when the observed/known "
                "availability and recipe constraints allow it."
            )

        if "webshop_page_state_mismatch" in error_types:
            guidance.append(
                "- WebShop page-state mismatches often come from parser schema drift. Split "
                "observations on `[SEP]`, strip every section, and expose canonical fields "
                "`page_type`, `current_page_number`, `selected_filters`, and active `asin`. "
                "Map search/results/product/purchase pages to stable page_type values."
            )
            guidance.append(
                "- For WebShop readout, preserve visible instruction/navigation/ASIN/title/price "
                "from the current belief. Do not emit fake totals or placeholder prices when "
                "details are unknown; render the known page skeleton instead."
            )

        return "\n".join(guidance)

    def _diagnose_failures(self, errors: List[Dict[str, Any]]) -> Dict[str, Any]:
        clusters: Dict[str, Dict[str, Any]] = {}
        for err in errors:
            error_type = str(err.get("error_type", "unknown") or "unknown")
            action_type = _action_type(str(err.get("action", "") or ""))
            key = f"{error_type}|{action_type}"
            cluster = clusters.setdefault(
                key,
                {
                    "error_type": error_type,
                    "action_type": action_type,
                    "count": 0,
                    "examples": [],
                    "priority": self._failure_priority(error_type),
                },
            )
            cluster["count"] += 1
            if len(cluster["examples"]) < 3:
                cluster["examples"].append(
                    {
                        "action": err.get("action", ""),
                        "expected": str(err.get("expected", ""))[:500],
                        "got": str(err.get("got", ""))[:500],
                        "observation": str(err.get("observation", ""))[:700],
                        "next_observation": str(err.get("next_observation", ""))[:700],
                        "predicted_observation": str(err.get("predicted_observation", ""))[:700],
                    }
                )

        ordered = sorted(
            clusters.values(),
            key=lambda item: (item["priority"], -item["count"], item["error_type"]),
        )
        return {
            "total_failures": len(errors),
            "cluster_count": len(ordered),
            "clusters": ordered[:12],
        }

    def _failure_priority(self, error_type: str) -> int:
        lowered = (error_type or "").lower()
        if "load" in lowered:
            return 0
        if "exception" in lowered or lowered.endswith("_error"):
            return 1
        if "missing_latent" in lowered:
            return 2
        if "mismatch" in lowered:
            return 3
        return 4

    def _prioritize_failures(self, errors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return sorted(
            errors,
            key=lambda err: (
                self._failure_priority(str(err.get("error_type", ""))),
                _action_type(str(err.get("action", ""))),
            ),
        )

    def _score_formal_test_result(self, errors: List[Dict[str, Any]]) -> float:
        if not errors:
            return 0.0
        if self._has_load_error(errors):
            return TRACEPATCH_LOAD_ERROR_SCORE + len(errors)
        score = 0.0
        for err in errors:
            error_type = str(err.get("error_type", "unknown"))
            score += 1.0 + (0.25 * self._failure_priority(error_type))
        return score

    def _has_load_error(self, errors: List[Dict[str, Any]]) -> bool:
        return any(str(err.get("error_type", "")) == "load_error" for err in errors)

    def _patch_improves(
        self,
        *,
        current_score: float,
        current_errors: List[Dict[str, Any]],
        candidate_score: float,
        candidate_errors: List[Dict[str, Any]],
        load_error_acceptance_cap: int = 1000,
    ) -> Tuple[bool, str]:
        current_load_error = self._has_load_error(current_errors)
        candidate_load_error = self._has_load_error(candidate_errors)
        if candidate_load_error and not current_load_error:
            return False, "candidate_load_error"
        if current_load_error and not candidate_load_error:
            if len(candidate_errors) > load_error_acceptance_cap:
                return False, "load_fix_too_many_replay_errors"
            return True, "fixes_load_error"
        if candidate_score < current_score:
            return True, "lower_formal_test_score"
        if candidate_score == current_score and len(candidate_errors) < len(current_errors):
            return True, "same_score_fewer_failures"
        return False, "not_better_than_current"

    def _format_transitions(self, transitions: List[Transition]) -> str:
        parts: List[str] = []
        for idx, step in enumerate(transitions, start=1):
            task_text = _task_text_from_observation(step.observation)
            task_family = _sciworld_task_family(step.observation) if task_text else ""
            entry = (
                f"[{idx}] ACT: {step.action}\n"
                f"    ACT_TYPE: {_action_type(step.action)}\n"
                f"    OUTCOME_TYPE: {_outcome_type(step.next_observation)}\n"
            )
            if task_text:
                entry += f"    TASK: {task_text[:220]}\n"
            if task_family and task_family != "other":
                entry += f"    TASK_FAMILY: {task_family}\n"
            entry += (
                f"    OBS: {(step.observation or '')[:700]}\n"
                f"    NEXT: {(step.next_observation or '')[:700]}"
            )
            parts.append(entry)
        return "\n\n".join(parts)

    def _format_errors(self, errors: List[Dict[str, Any]]) -> str:
        parts: List[str] = []
        for err in errors:
            lines = [
                f"Step: action='{err.get('action', '')}'",
                f"  Error: {err.get('error_type', '')}",
                f"  Expected fields: {str(err.get('expected', ''))[:1000]}",
                f"  Got fields: {str(err.get('got', ''))[:1000]}",
            ]
            observation = str(err.get("observation", "") or "")
            next_observation = str(err.get("next_observation", "") or "")
            predicted_observation = str(err.get("predicted_observation", "") or "")
            if observation:
                lines.append(f"  OBS: {observation[:1500]}")
            if next_observation:
                lines.append(f"  EXPECTED_NEXT: {next_observation[:1500]}")
            if predicted_observation:
                lines.append(f"  PREDICTED_OBS: {predicted_observation[:1500]}")
            parts.append("\n".join(lines))
        return "\n\n".join(parts)

    def _summarize_errors(self, errors: List[Dict[str, Any]]) -> str:
        counts: Dict[str, int] = {}
        for err in errors:
            key = str(err.get("error_type", "unknown")).split(" ", 1)[0]
            counts[key] = counts.get(key, 0) + 1
        if not counts:
            return "No validation errors."
        return "\n".join(f"- {key}: {count}" for key, count in sorted(counts.items()))

    def _format_diagnosis(self, diagnosis: Dict[str, Any]) -> str:
        clusters = diagnosis.get("clusters", [])
        if not clusters:
            return "No validation errors."
        lines = [
            f"Total failures: {diagnosis.get('total_failures', 0)}",
            f"Failure clusters: {diagnosis.get('cluster_count', len(clusters))}",
        ]
        for idx, cluster in enumerate(clusters, start=1):
            lines.append(
                f"{idx}. {cluster.get('error_type', 'unknown')} "
                f"action_type={cluster.get('action_type', '') or '<none>'} "
                f"count={cluster.get('count', 0)}"
            )
            examples = cluster.get("examples", [])
            for example in examples[:2]:
                lines.append(
                    "   - "
                    f"action={example.get('action', '')!r}; "
                    f"expected={example.get('expected', '')!r}; "
                    f"got={example.get('got', '')!r}"
                )
                if example.get("next_observation") or example.get("predicted_observation"):
                    lines.append(
                        "     raw_next="
                        f"{example.get('next_observation', '')!r}; "
                        "raw_pred="
                        f"{example.get('predicted_observation', '')!r}"
                    )
        return "\n".join(lines)

    def _environment_guidance(self, env_name: str) -> str:
        return ENV_PROMPT_GUIDANCE.get((env_name or "").strip().lower(), "")

    def _track_guidance(self) -> str:
        if self.induction_track != "epistemic_state":
            return ""
        return (
            "Use an explicit epistemic belief schema instead of a flat observation cache. "
            "Represent directly observed facts separately from latent variables, frontier/unseen "
            "entities, and candidate hypotheses. When multiple hidden states remain possible, "
            "use bounded set-valued or probabilistic fields such as `posterior`, `candidate_probs`, "
            "`object_location_beliefs`, or weighted `hypotheses`; normalize probabilities when "
            "straightforward. Prediction/readout should remain deterministic for evaluation: render "
            "from the most likely or most constrained hypothesis, and preserve uncertainty in the "
            "belief for later correction rather than sampling random observations."
        )

    # ------------------------------------------------------------------
    # Compatibility helpers retained for older tests/configs.
    # ------------------------------------------------------------------

    def _prompt_sampling_params(
        self,
        env_name: str,
        trajectories: List[Trajectory],
    ) -> Tuple[int, int, int, int]:
        action_types: set[str] = set()
        patterns: set[Tuple[str, str]] = set()
        for traj in trajectories:
            for step in traj.transitions:
                action_type = _action_type(step.action)
                outcome_type = _outcome_type(step.next_observation)
                action_types.add(action_type)
                patterns.add((action_type, outcome_type))

        budget = self.contrastive_max_transitions
        max_per_pattern = self.contrastive_max_per_pattern
        if (env_name or "").strip().lower() == "sciworld" and len(action_types) > budget:
            budget = min(640, max(budget, len(action_types) * 2))
            max_per_pattern = max(max_per_pattern, 6)
        return budget, max_per_pattern, len(action_types), len(patterns)

    def _compose_selection_score(
        self,
        *,
        predictive_loss: float,
        abstraction_loss: float,
        validity_loss: float,
    ) -> float:
        return (
            float(predictive_loss)
            + 0.10 * float(abstraction_loss)
            + 0.10 * float(validity_loss)
        )

    def _objective_terms(
        self,
        *,
        base_loss: float,
        coverage: float,
        exact_match_rate_all: float,
        replay_penalty: float,
        obs_error_ratio: float,
        invariant_violation_ratio: float,
        abstraction_gap_ratio: float,
        placeholder_penalty: float,
        placeholder_violation_rate: float,
        parse_failure_rate: float,
        invariant_penalty: float,
    ) -> Tuple[float, float, float]:
        predictive_loss = (
            float(base_loss)
            + 0.25 * max(0.0, 1.0 - float(coverage))
            + 0.10 * max(0.0, 1.0 - float(exact_match_rate_all))
            + float(replay_penalty)
            + 0.10 * float(obs_error_ratio)
        )
        abstraction_loss = float(abstraction_gap_ratio)
        validity_loss = (
            0.50 * float(invariant_violation_ratio)
            + float(placeholder_penalty)
            + float(placeholder_violation_rate)
            + float(parse_failure_rate)
            + float(invariant_penalty)
        )
        return predictive_loss, abstraction_loss, validity_loss

    def _selection_improves(
        self,
        env_name: str,
        current: Dict[str, Any],
        candidate: Dict[str, Any],
    ) -> Tuple[bool, str]:
        del env_name
        current_replay = int(current.get("replay_error_count", 10**9))
        candidate_replay = int(candidate.get("replay_error_count", 10**9))
        if candidate_replay > current_replay:
            return False, "worse_replay_errors"
        if candidate_replay < current_replay:
            return True, "fewer_replay_errors"

        current_predictive = float(current.get("predictive_loss", float("inf")))
        candidate_predictive = float(candidate.get("predictive_loss", float("inf")))
        if candidate_predictive < current_predictive:
            return True, "better_predictive_loss"
        if candidate_predictive > current_predictive:
            return False, "worse_predictive_loss"

        return (
            float(candidate.get("score", float("inf")))
            < float(current.get("score", float("inf"))),
            "better_score",
        )

    def _generate_best_initial_code(
        self,
        env_name: str,
        is_spatial: bool,
        class_name: str,
        rules: str,
        transition_text: str,
        validation_trajectories: List[Trajectory],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], int]:
        n_initial_candidates = max(1, int(self._initial_candidate_count(env_name)))
        prompt = self._build_coding_prompt(
            env_name,
            is_spatial,
            class_name,
            rules,
            transition_text,
        )
        initial_candidates: List[Dict[str, Any]] = []
        for candidate_idx in range(1, n_initial_candidates + 1):
            code = self._extract_code(self.llm.generate(prompt))
            candidate = self._evaluate_candidate(
                code,
                env_name,
                validation_trajectories,
                label=f"initial_candidate_{candidate_idx:02d}",
            )
            initial_candidates.append(candidate)

        initial_candidates.sort(
            key=lambda item: (
                bool(item.get("hard_reject", False)),
                float(item.get("score", float("inf"))),
                int(item.get("replay_error_count", 10**9)),
            )
        )
        beam = initial_candidates[: self.patch_beam_size]
        if not beam and initial_candidates:
            beam = [initial_candidates[0]]
        return beam, initial_candidates, n_initial_candidates

    def _evaluate_candidate(
        self,
        code: str,
        env_name: str,
        validation_trajectories: List[Trajectory],
        *,
        label: str,
    ) -> Dict[str, Any]:
        del env_name
        errors = self._run_formal_tests(code, validation_trajectories, label=label)
        score = self._score_formal_test_result(errors)
        return {
            "code": code,
            "loadable": not any(err.get("error_type") == "load_error" for err in errors),
            "hard_reject": False,
            "score": score,
            "predictive_loss": score,
            "abstraction_loss": 0.0,
            "validity_loss": 0.0,
            "error_count": len(errors),
            "replay_error_count": len(errors),
            "obs_error_count": len(errors),
            "base_loss": score,
            "failure_mode": self._summarize_errors(errors),
        }

    def _initial_candidate_count(self, env_name: str) -> int:
        del env_name
        return 1

    def _max_hard_reject_retries(self, env_name: str) -> int:
        del env_name
        return 0

    def _search_guidance(self, *args: Any, **kwargs: Any) -> str:
        del args, kwargs
        return ""

    def _bootstrap_prior_initial_candidate(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        return None

    def _extract_code(self, llm_output: str) -> str:
        text = (llm_output or "").strip()
        if "```python" in text:
            return text.split("```python", 1)[1].split("```", 1)[0].strip()
        if "```" in text:
            return text.split("```", 1)[1].split("```", 1)[0].strip()
        return text

    def _save_model_snapshot(
        self,
        *,
        env_name: str,
        phase: str,
        code: str,
        metadata: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        if self.snapshot_dir is None:
            return None
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self._snapshot_counter += 1
        prefix = f"{self._snapshot_counter:03d}_{phase}"
        code_path = self.snapshot_dir / f"{prefix}.py"
        meta_path = self.snapshot_dir / f"{prefix}.json"
        code_path.write_text(code, encoding="utf-8")
        payload = {
            "env": env_name,
            "phase": phase,
            **metadata,
            "code_path": str(code_path),
        }
        meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        record = {
            "phase": phase,
            "code_path": str(code_path),
            "metadata_path": str(meta_path),
        }
        self._version_records.append(record)
        return record

    def _progress(self, message: str) -> None:
        if self.verbose or self.show_progress:
            print(f"[patchworld_inducer] {message}")
