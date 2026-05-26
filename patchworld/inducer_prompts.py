"""LLM prompt templates and per-environment guidance for world-model induction."""

from typing import Dict, List


SPATIAL_RULE_EXTRACTION_PROMPT = """You are analysing trajectories from the SPATIAL environment: {env_name}.

Each trajectory shows (OBS, ACT, NEXT_OBS) tuples from different problem instances.

{transition_examples}

Extract the GENERAL transition rules of this environment. For each rule, your answer must generalise
across ALL problem instances — do not mention specific object names, room names, or coordinates.

Answer in this exact format:

## Observation format
(Describe the structural template of observations — what fields always appear, what delimiters,
what varies vs what is fixed. Give a generic template like "Room: <name>\\nObjects: <list>".)

## Action types and syntax
(List every action type seen, with its generic syntax. E.g. "go to <location>", "pick up <object>".)

## Transition rules
(For each action type, state: given current state S and action A, what does NEXT_OBS look like?
Express as a function of S and A — no specific values. E.g.
  "go to <loc>: if loc is connected to current room and door is open → agent moves to loc,
   NEXT_OBS = [room description of loc with its objects]".)

## Failure conditions
(What causes an action to produce a failure response? What does the failure response look like?)

## Belief state design
(What fields should the belief state track to support predict_observation?
 E.g. agent_location, room_graph, object_locations, door_states, inventory, goal.)
"""

API_RULE_EXTRACTION_PROMPT = """You are analysing trajectories from the environment: {env_name}.

Each trajectory shows (OBS, ACT, NEXT_OBS) tuples from different problem instances.

{transition_examples}

Extract the GENERAL transition rules of this environment. For each rule, your answer must generalise
across ALL problem instances — do not mention specific product names, search queries, or task values.

Answer in this exact format:

## Observation format
(Describe the structural template — what delimiters, what sections, what fields always appear.
 Give a generic template like "WebShop [SEP] Instruction: [SEP] <goal> [SEP] <page_content>".)

## Action types and syntax
(List every action type with generic syntax. E.g. "search[<query>]", "click[<element>]".)

## Transition rules
(For each action type, state: given current state S and action A, what does NEXT_OBS look like?
 Express as a function of S and A. E.g.
  "search[q]: NEXT_OBS = Instruction: [SEP] <goal> [SEP] Back to Search [SEP] Page 1 ...")

## Failure conditions
(What causes an action to fail? What does the failure response look like?)

## Belief state design
(What fields should the belief state track? E.g. page_type, instruction, available_actions, cart.)
"""

# ---------------------------------------------------------------------------
# Stage 2: Coding prompts (receive extracted rules + contrastive examples)
# ---------------------------------------------------------------------------

SPATIAL_CODING_PROMPT = """You are implementing a Python world model for the SPATIAL environment: {env_name}.

## Extracted transition rules
{rules}

## Contrastive example transitions
These examples were chosen to show the SAME action type with DIFFERENT inputs/outputs,
so you can see the parametric pattern:

{transition_examples}

Implement the world model class following the rules above exactly.
The class must work on NEW problems with different object names, room layouts, and goals.

Treat each observation as PARTIAL EVIDENCE about the hidden world, not as the full environment state.
The model should maintain a belief state that combines:
- directly observed facts from the current observation, and
- latent hypotheses / candidate hidden state needed to predict future observations.
- epistemic status for each tracked fact (`known`, `unknown`, `unobserved`, `constrained`, etc.).

Keep the belief lightweight, but do include hidden-state fields whenever the environment is partially
observed (for example unknown object locations, candidate answers, unseen page state, or latent map info).
Do not build a huge symbolic graph unless the trajectory evidence clearly requires it.

Important design constraints for this project:
- Never convert missing evidence into negative facts (do NOT infer "empty" when content is merely unseen).
- Distinguish information-revealing actions (look/open/enter/search) from state-changing actions.
- Use a graceful renderer cascade: exact template when known, constrained rendering when partial, text-level fallback when uncertain.
- When prediction mismatches reality, use abductive reconciliation to update latent beliefs or schema assumptions.

```python
from patchworld.worldmodel_base import BaseWorldModel
# Optional helper scaffold:
# from patchworld.epistemic_worldmodel import (
#     EpistemicWorldModel, BeliefState, EpistemicStatus, TransitionType
# )

class {class_name}(BaseWorldModel):
    def init_belief(self) -> dict:
        return {{}}

    def init_belief_from_observation(self, obs_text: str) -> dict:
        belief = self.init_belief()
        return self.correct_belief(belief, obs_text, "")

    def predict_belief(self, belief: dict, action: str) -> dict:
        # Predict step only (no observation correction): b' = f(b, a)
        # Apply action-conditioned dynamics and return prior belief.
        ...

    def correct_belief(self, belief_prior: dict, obs_text: str) -> dict:
        # Correction step: b = g(b', o)
        # Assimilate actual observation evidence into prior belief.
        ...

    def readout_observation(self, belief: dict, action: str = "") -> str:
        # Readout authority.
        # Reuse the same template/sections that parse_observation expects.
        # parse_observation(readout_observation(...)) should recover visible fields implied by belief.
        # Must always return a non-empty string.
        ...

    def parse_observation(self, obs_text: str) -> dict:
        # Return directly observed evidence / visible fields (reuse correction parsing logic).
        ...

    # Backward-compatible wrappers expected by existing evaluators.
    def transition(self, belief: dict, action: str) -> dict:
        return self.predict_belief(belief, action)

    def predict_next(self, state: dict, action: str) -> dict:
        return self.predict_belief(state, action)

    def predict_observation(self, belief: dict, action: str) -> str:
        belief_next_prior = self.predict_belief(belief, action)
        return self.readout_observation(belief_next_prior, action)

    def extract_valid_action_forms(self) -> dict[str, list[str]]:
        ...
```

Hard constraints:
- NEVER hardcode specific values from the examples (object names, coordinates, product names, etc.).
- Treat the observation as partial evidence and maintain latent belief when hidden state matters.
- correct_belief must handle any new problem instance, not just the training ones.
- Implement canonical predict/correct/readout pipeline:
  init_belief_from_observation(o1) -> predict_belief(b, a) -> readout_observation(b') -> correct_belief(b', o_next).
- Keep transition/predict_next/predict_observation as thin compatibility wrappers around predict_belief/readout_observation.
- Primary prediction quality should come from pre-correction readout (readout of b').
- predict_observation must return a non-empty string on every call.
- Reuse the real observation template from the examples instead of inventing new prose.

Output ONLY the Python code, nothing else.
"""

API_CODING_PROMPT = """You are implementing a Python world model for the environment: {env_name}.

## Extracted transition rules
{rules}

## Contrastive example transitions
These examples were chosen to show the SAME action type with DIFFERENT inputs/outputs,
so you can see the parametric pattern:

{transition_examples}

Implement the world model class following the rules above exactly.
The class must work on NEW problems with different products, queries, words, or tasks.

Treat each observation as PARTIAL EVIDENCE about the hidden world, not as the full environment state.
The belief should keep directly observed page/text structure plus latent fields or hypotheses whenever
future observations depend on hidden state (candidate answers, page identity, hidden product details, etc.).
Keep the belief lightweight, but do not avoid latent state when it is necessary for prediction.

Important design constraints for this project:
- Attach epistemic status to tracked facts (known/inferred/constrained/unknown/unobserved).
- Treat information-revealing actions separately from world-mutating actions.
- Preserve raw text buffers so parser failures do not erase usable evidence.
- If predicted observation diverges from actual, reconcile by abductively updating latent variables/rules.

```python
from patchworld.worldmodel_base import BaseWorldModel
# Optional helper scaffold:
# from patchworld.epistemic_worldmodel import (
#     EpistemicWorldModel, BeliefState, EpistemicStatus, TransitionType
# )

class {class_name}(BaseWorldModel):
    def init_belief(self) -> dict:
        return {{}}

    def init_belief_from_observation(self, obs_text: str) -> dict:
        belief = self.init_belief()
        return self.correct_belief(belief, obs_text, "")

    def predict_belief(self, belief: dict, action: str) -> dict:
        # Predict step only (no observation correction): b' = f(b, a)
        ...

    def correct_belief(self, belief_prior: dict, obs_text: str) -> dict:
        # Correction step: b = g(b', o)
        ...

    def readout_observation(self, belief: dict, action: str = "") -> str:
        # Readout authority.
        # Fill the observation FORMAT template using latent belief state values.
        # Keep delimiters/section order consistent with what parse_observation expects.
        # Must always return a non-empty string.
        ...

    def parse_observation(self, obs_text: str) -> dict:
        ...

    # Backward-compatible wrappers expected by existing evaluators.
    def transition(self, belief: dict, action: str) -> dict:
        return self.predict_belief(belief, action)

    def predict_next(self, state: dict, action: str) -> dict:
        return self.predict_belief(state, action)

    def predict_observation(self, belief: dict, action: str) -> str:
        belief_next_prior = self.predict_belief(belief, action)
        return self.readout_observation(belief_next_prior, action)

    def extract_valid_action_forms(self) -> dict[str, list[str]]:
        ...
```

Hard constraints:
- NEVER hardcode specific values from the examples.
- Treat the observation as partial evidence and maintain latent belief when hidden state matters.
- correct_belief must handle any new problem instance.
- Implement canonical predict/correct/readout pipeline:
  init_belief_from_observation(o1) -> predict_belief(b, a) -> readout_observation(b') -> correct_belief(b', o_next).
- Keep transition/predict_next/predict_observation as thin compatibility wrappers around predict_belief/readout_observation.
- Primary prediction quality should come from pre-correction readout (readout of b').
- predict_observation must return a non-empty string on every call.
- Reuse the real observation template from the examples instead of inventing new prose.

Output ONLY the Python code, nothing else.
"""

EPISTEMIC_RULE_EXTRACTION_PROMPT = """You are analyzing trajectories from the environment: {env_name}.

Each transition comes from a partially observed environment, so the induced rules must separate:
- what is directly observed now,
- what remains latent or uncertain,
- what should be carried forward in belief state,
- and what exact observation template the renderer must preserve.

{transition_examples}

Extract GENERAL epistemic transition rules that transfer across all task instances.
Do not mention concrete object names, room names, ASINs, queries, or trajectory-specific literals.

Answer in this exact format:

## Observation schema invariants
(Describe the exact visible template: fixed delimiters/sections, required fields, failure formats, and how page/room/inventory/task text is structured.)

## Latent state and epistemic variables
(List the hidden or partially observed variables that must be tracked, plus explicit known/unknown/frontier/constrained status fields.)

## Action families
(List each action family with canonical syntax and whether it is state-changing, information-revealing, or both.)

## Transition rules
(For each action family, describe how visible state changes, how latent state changes, and what the next observation should look like.)

## Uncertainty handling
(Explain how to update priors/hypotheses/frontier when the observation is ambiguous or incomplete.)

## Rendering constraints
(Specify strict observation-format constraints the renderer must preserve, and which fields must never be hallucinated.)
"""

EPISTEMIC_CODING_PROMPT = """You are implementing a Python EPISTEMIC world model for the environment: {env_name}.

## Extracted transition rules
{rules}

## Contrastive example transitions
{transition_examples}

Implement a standalone epistemic-state world model class. This model must be independent from simple_state.
It should model uncertainty explicitly and support abductive reconciliation.
This epistemic model MUST remain pure symbolic (no external model calls at runtime).
Treat extracted rules as strong defaults, but allow uncertainty-aware exceptions when evidence is sparse or conflicting.

Design requirements:
- Preserve raw observation text in the belief state for non-lossy fallback rendering.
- Track structured facts with epistemic status labels (`known`, `inferred`, `constrained`, `unknown`, `unobserved`, `default`).
- Maintain latent variables for hidden causes when outcomes vary under similar visible state.
- Keep an open-world frontier for entities/locations that are mentioned but not fully observed.
- Distinguish information-revealing actions from state-changing actions.
- Keep action-conditioned history summaries (counts/frequencies) of past outcomes in belief state.
- Maintain probability-like scores over hypotheses (e.g., room-content priors) and update them from history.
- For unseen/unknown contexts (such as an unvisited room), infer likely possibilities from prior trajectories/history rather than defaulting to empty state.
- Keep parser/update/predict/render internally consistent:
  parse_observation(predict_observation(...)) should recover the visible state implied by predict_next.
- Add an abductive update path that reconciles predicted vs actual observations.

Use/extend this scaffold style:
```python
from patchworld.worldmodel_base import BaseWorldModel
# Optional helper scaffold:
# from patchworld.epistemic_worldmodel import (
#     BeliefState, EpistemicStatus, TransitionType
# )

class {class_name}(BaseWorldModel):
    def init_belief(self):
        ...

    def init_belief_from_observation(self, obs_text: str):
        belief = self.init_belief()
        return self.correct_belief(belief, obs_text, "")

    def parse_observation(self, obs_text: str) -> dict:
        ...

    def predict_belief(self, belief, action: str):
        # Predict step only (no correction): b' = f(b, a)
        ...

    def correct_belief(self, belief_prior, obs_text: str):
        # Correction step: b = g(b', o)
        ...

    def readout_observation(self, belief, action: str = "") -> str:
        # Implement renderer cascade: template when known, constrained when partial, text fallback when uncertain.
        # Must always return a non-empty string.
        ...

    # Backward-compatible wrappers expected by existing evaluators.
    def transition(self, belief, action: str):
        return self.predict_belief(belief, action)

    def predict_next(self, state: dict, action: str) -> dict:
        return self.predict_belief(state, action)

    def predict_observation(self, belief, action: str) -> str:
        belief_next_prior = self.predict_belief(belief, action)
        return self.readout_observation(belief_next_prior, action)

    def extract_valid_action_forms(self) -> dict[str, list[str]]:
        ...
```

Core constraints:
- NEVER hardcode trajectory-specific literals.
- Keep epistemic uncertainty explicit; do not collapse unobserved/unknown to empty defaults.
- Keep model lightweight and robust to new tasks.
- Implement canonical predict/correct/readout pipeline:
  init_belief_from_observation(o1) -> predict_belief(b, a) -> readout_observation(b') -> correct_belief(b', o_next).
- Keep transition/predict_next/predict_observation as thin compatibility wrappers around predict_belief/readout_observation.
- Primary prediction quality should come from pre-correction readout (readout of b').
- predict_observation must return a non-empty string on every call.
- Pure symbolic only: do NOT import/use `openai`, `anthropic`, `requests`, `httpx`, sockets, subprocesses, or any remote/API client.
- Do NOT call `.generate(...)`, `.chat(...)`, or any external LLM from induced model code.
- Generated models must be fully self-contained: do NOT read from or write to local files, sidecar data files, temp files, package data, or absolute paths.
- Prefer deterministic outputs by selecting the most likely hypothesis when multiple outcomes are plausible, while retaining alternative hypotheses in belief for future updates.

Trajectory-grounded environment checks:
- `alfworld`: preserve object/receptacle visibility transitions; model "Nothing happens." as action-conditional, not global fallback.
- `babyai`: keep goal text, carrying state, available-actions list shape, and orientation/wall cues consistent.
- `maze`: preserve wall directions in every non-terminal observation and keep coordinate movement deterministic.
- `sciworld`: model room inventory/container/door/open states; avoid generic fallback strings unless evidence supports them.
- `textcraft`: parse command recipes + inventory and make `get`/`craft` update counts deterministically.
- `webshop`: preserve `[SEP]` page structure, ASIN/page/filter state, and action-conditioned page transitions.
- `wordle`: maintain candidate-set constraints from guess/feedback history; never emit constant feedback for all guesses.

Output ONLY the Python code, nothing else.
"""

ENV_PROMPT_GUIDANCE: Dict[str, str] = {
    "wordle": """
- Treat all five-letter guesses as the SAME action family; do not memorize specific guess strings.
- Never use a tiny hardcoded dictionary or toy word list.
- Never return a placeholder default like "b b b b b" for all valid guesses.
- Maintain enough latent state from previous feedback to narrow candidate answers and generate feedback that is consistent with the observed history.
- Use a canonical explicit belief that tracks at least `guess_history`, `feedback_history`, and either `possible_words` or another candidate-set representation.
- Parse and emit Wordle feedback exactly as five space-separated symbols from {b,y,g}, e.g. "b b y g b".
- If the environment observation exposes the admissible guess vocabulary, use that vocabulary rather than inventing words.
- Do not load a word list from disk or require `wordle_words.txt`; keep any vocabulary/candidate representation inside the Python code and learned belief state.
""".strip(),
    "webshop": """
- Preserve the exact [SEP]-delimited observation structure.
- When parsing, split on `[SEP]` and strip whitespace from every section; real observations use `WebShop [SEP] Instruction:` with spaces around the delimiter.
- `parse_observation` must expose evaluator-recognized canonical fields: `page_type`, `current_page_number`, `selected_filters`, and either `asin`/`product_details["asin"]` or `page_content` containing visible ASINs.
- Never emit placeholder values such as "Total results: 50", "$XX.XX", "option1", or generic fake product details.
- Reuse concrete navigation controls, page markers, ASINs, titles, filters, and product fields parsed from the current observation whenever the next page depends on them.
- Distinguish search-results pages, filter-option pages, and product-detail pages using the actual observed structure.
- Track a canonical explicit page state with fields such as `page_type`, `current_page_number`, `selected_filters`, and the concrete active/purchased ASIN when available.
- For `click[<ASIN>]`, copy the clicked ASIN plus any visible title/price from the current results list into product state before rendering the product page.
- For `click[Back to Search]`, render the actual search page template: `WebShop [SEP] Instruction: [SEP] <instruction> [SEP] Search`.
- For unknown product details, prefer a conservative page with known ASIN/title/navigation over fake prices or fabricated attributes.
- Only predict failure messages when they match the environment's real failure patterns.
""".strip(),
    "sciworld": """
- Do not fall back to generic responses such as "No known action matches that input" unless the trajectory evidence shows that exact failure pattern.
- Track room contents, inventory, object/container state, and measurable properties so predictions stay grounded in observed state.
- Keep an explicit state that includes at least current room, inventory, and key door/container/measurement fields needed to replay the task family.
- Prefer conservative state-preserving predictions over generic placeholder text.
- Keep observation structure faithful to the environment's actual room/inventory/measurement format.
""".strip(),
    "alfworld": """
- Reuse the environment's observed room/object/inventory template instead of inventing a new narration style.
- Keep room names, receptacle listings, and inventory formatting parser-consistent across correct_belief and render_observation.
- Track an explicit state that includes current location, visible receptacles/objects, receptacle open/closed state, and inventory.
- If an action should preserve visible state, preserve the prior room/object structure rather than returning a generic fallback.
""".strip(),
    "babyai": """
- Track an explicit state for visible objects, carried item(s), wall distance/orientation cues, and task goal instead of only rewriting the text surface.
- Preserve object and door identities across `correct_belief`, `transition`, and `render_observation`.
- When navigation changes what is visible, update the structured state first and render from that state.
""".strip(),
    "textcraft": """
- Parse inventory and resource counts explicitly from observations instead of leaving inventory empty.
- `get` and `craft` must change state in a way that is consistent with materials, recipes, and resource counts.
- Never return generic craft success without consuming ingredients or updating the resulting item counts.
- Keep a canonical explicit inventory-count state plus recipe and goal fields so `transition` and `render_observation` stay consistent.
- Preserve the exact inventory / crafting text structure used by the environment.
- Treat TextCraft as partially observed: the belief must include recognized latent support fields such as `latent_variables`, `facts`, `hypotheses`, `frontier`, or `hidden_state`; a belief containing only `inventory`, `recipes`, and raw text is not enough.
- Use evaluator-recognized field names where possible: `crafting_recipes` for known recipe structures, `goal_item` for the parsed goal target, and `last_action_outcome` for `got`, `crafted`, `not_found`, `invalid_recipe`, or `inventory` outcomes.
- Track hidden availability separately from inventory, for example `latent_variables = {"obtainable_items": set(), "known_unavailable_items": set(), "recipe_graph": {...}}`, and update it from `Could not find ...` / `Could not find a valid recipe ...` observations.
- Parse inventory entries in the real format `[item] (count)` and keep item-name normalization consistent between inventory, recipes, actions, and rendered observations.
- `get <count> <item>` should not always succeed: if the item is known unavailable, render the observed failure format `Could not find <item>` and avoid changing inventory.
- `readout_observation` should render action-specific outcome text using the action arguments, e.g. `Got <count> <item>` and `Crafted <count> minecraft:<item>`; avoid generic strings like `Got item` or `Crafted item`.
- Keep both `recipes` and `crafting_recipes` synchronized when recipe structures are parsed so state scoring can see known recipe counts.
""".strip(),
    "maze": """
- Distinguish successful movement from blocked movement; wall/boundary moves should preserve the current state instead of hallucinating a new room.
- Keep movement logic action-conditioned: north/south/east/west should not collapse to the same generic response.
- Reuse the observed maze/grid format exactly instead of inventing new room descriptions.
- Parse the CURRENT maze state from the final environment block, not the instructional example text that appears earlier in some observations.
- Use one canonical explicit state schema across `parse_observation`, `correct_belief`, `transition`, and `render_observation`: `agent_x`, `agent_y`, `goal_x`, `goal_y`, `local_walls`, and optional `status`.
- Parse all visible wall directions into `local_walls`, and render those wall directions back on every non-terminal next observation.
- Use canonical wall directions internally: `up`, `down`, `left`, `right`. Accept aliases like `above you` / `below you` only at parse/render boundaries.
- Maintain latent map memory such as `wall_map[(agent_x, agent_y)] -> set(walls)` and `visited_cells`; `local_walls` alone is insufficient because movement prediction needs remembered wall evidence and destination-cell wall hypotheses.
- Do not set `status = success` merely because the instructional preamble contains the word `Success`; only exact terminal observations or actual goal-reaching transitions should set terminal status.
- Preserve canonical wall wording exactly: one wall => `There is a wall ...`; multiple walls => `There are walls ...` with deterministic direction ordering.
- Minimize paraphrase: keep static wording/punctuation identical to observed Maze templates and only change coordinates/wall clauses/status when state changes.
- For blocked moves, prefer near-copy rendering of the previous observation with unchanged coordinates and explicit wall/failure wording.
- `parse_observation(render_observation(...))` must recover the same maze state as `transition(...)`; omitting wall text is invalid even if coordinates are correct.
""".strip(),
}

ENV_PLACEHOLDER_PATTERNS: Dict[str, List[str]] = {
    "wordle": [
        "b b b b b",
        "minimal dictionary",
        "toy word list",
        "default feedback",
        "_load_dictionary",
        "can't predict feedback",
    ],
    "webshop": [
        "$xx.xx",
        "option1",
        "option2",
        "unknown",
        "in a real model",
    ],
    "sciworld": [
        "no known action matches that input",
        "i'm not sure how to use",
        "simplified prediction",
        "this would need more sophisticated",
        "for simplicity",
    ],
    "textcraft": [
        "crafting result",
        "inventory: empty",
        "you successfully craft",
        "generic inventory",
    ],
    "alfworld": [
        "nothing happens",
        "default response for navigation",
        "simplified version",
    ],
    "babyai": [
        "simulate going through door - replace all objects with empty room",
        "placeholder",
        "for simplicity",
    ],
    "maze": [
        "you move to a new position",
        "placeholder maze",
        "generic wall response",
        "without wall information",
        "don't have information about walls",
    ],
}

REFINE_PROMPT = """The world model you generated has problems when replayed against trajectories.

Current code:
{code}

Problems found:
{errors}

Failure bucket summary:
{error_summary}

Stage-1 transition rules and invariants:
{rules_guidance}

Fix the code so that:
1. All exceptions are resolved.
2. predict_observation always returns a non-empty string and its output structurally matches
   the actual next observations shown above (same delimiters, fields, format).
3. Treat each observation as partial evidence and maintain latent belief or candidate hidden state
   whenever the next observation depends on unobserved information.
4. parse_observation(predict_observation(...)) preserves the key visible state fields implied by the transition.
5. Prioritize replay failures and repeated bucket failures before stylistic wording.
6. Do NOT hardcode values from training examples — the fix must generalise.

Output the complete corrected Python code, nothing else.
"""

