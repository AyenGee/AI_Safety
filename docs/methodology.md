# Methodology

This document is intended to mirror the methodology chapter of the research
proposal ("Evaluating the Impact of Linear Temporal Logic Verification on
Recall-Safety Tradeoffs in Multi-Agent Intent Filtering for LLM-Enabled
Robots", George Ayensu, Wits University, supervisors Steven James and
Benjamin Rosman), and to record any point where the implementation deviates
from that design as the project progresses. It is a living document, updated
per phase (see the roadmap in [../README.md](../README.md)).

**This is a skeleton.** The full proposal text is not reproduced here to
avoid drift between two copies of the same content; sections below are
placeholders to be filled in / linked to the proposal directly. Do not treat
placeholder text as a citation - anything needing a literature reference is
marked `TODO(cite)` rather than invented.

## Research question

Does integrating Linear Temporal Logic (LTL) formal verification into a
modular multi-agent LLM intent-filtering layer improve rejection of
unsafe/misdirected robotic commands while preserving recall on legitimate
commands, compared to architectures without formal verification?

TODO: paste/link the full research question, hypotheses, and success
criteria from the proposal document.

## Systems under comparison

See [architecture.md](architecture.md) for the four systems (Single-LLM,
Multi-Agent Planner-Critic, Single-LLM+LTL, Multi-Agent+LTL) and their data
flow diagrams.

## Environment and domain model

Implemented in `intent_filter/environment/`. A symbolic household domain
(rooms, objects, roles, world variables) rather than a full 3D simulator for
v1 - see the README's "Environment simulation" note and
`intent_filter/environment/backend.py` for the `SimulatorBackend` interface
that would let this be swapped for VirtualHome/AI2-THOR later.

The planning problem tuple `P = <O, Pr, A, S, T, I, G, tau>`
(`intent_filter/environment/problem.py`) and the safety rule base
(`config/safety_rules.yaml`) are the ground truth against which the LTL
verifier checks candidate plans.

## LTL vs. LTLf: formalism choice

The proposal's notation uses infinite-trace LTL operators (`G`, `F`, `U`).
In practice, this system evaluates **finite** robot command sequences - a
command completes (or is rejected) rather than running forever - which is
the domain LTLf (finite-trace LTL) is designed for.

**Decision: LTLf via `flloat`.** Three candidates were evaluated directly
against this repository's actual (Windows) development environment, not
just on paper:

- `spot` - the canonical LTL-to-Buchi-automaton toolkit, but it has **no
  PyPI distribution at all** (`pip index versions spot` returns no match on
  Windows); it is distributed via conda-forge or built from source, and is
  Linux/macOS-oriented. Adopting it would mean requiring contributors to run
  this research code inside WSL or a conda environment - a real setup cost
  for a single-machine student project - for a formalism (infinite-trace
  LTL) that is arguably the wrong fit anyway (see below).
- `ltlf2dfa` - LTLf-native, but its DFA translation shells out to the
  external MONA binary, which is its own non-trivial Windows install.
- `flloat` - LTLf-native, pure Python, installs via plain `pip install
  flloat` with no external binaries (verified: `pip install flloat`
  succeeds cleanly on Windows, pulling in only pure-Python deps -
  `pythomata`, `lark-parser`, `sympy`). It evaluates formula truth directly
  over a finite trace (`formula.truth(trace, 0)`), which is exactly the
  finite-horizon semantics this domain needs, without requiring full
  automaton construction for the simple checking task at hand.

`flloat` was adopted for both practical (cross-platform, zero external
binaries) and methodological (finite-horizon domain fit) reasons. The
safety-rule YAML (`config/safety_rules.yaml`) keeps standard LTL notation
(`G`, `F`, `U`, `X`) since finite-trace and infinite-trace operators share
syntax and differ only in semantics over finite vs. infinite traces - so the
rule base reads the same way the proposal describes it, while `flloat`
supplies LTLf semantics underneath.

**Implementation note - atom name sanitization.** `flloat`'s grammar treats
parentheses purely as grouping syntax, so the proposal's function-style atom
names (e.g. `agent_at(child_room)`, `has_object(knife)`) are not valid
`flloat` atom tokens as written - `agent_at(child_room)` parses as atom
`agent_at` followed by an unexpected `(`. Rather than flattening the rule
base's syntax (which would make `config/safety_rules.yaml` less readable and
diverge further from the proposal's notation), `intent_filter/verifier/atoms.py`
builds a fixed mapping from every grounded atom in the ontology (per room,
object, and role) to a flat identifier (e.g. `agent_at__child_room`),
applied consistently to both the formula string and the AP trace before
parsing, and reversed when building human-readable violation explanations.
This is transparent to rule authors and to the rest of the pipeline; only
`intent_filter/verifier` needs to know about it.

## Agent design notes (Phase 4)

The Planner, Critic, and NL->LTL Translator (`intent_filter/agents/`) all
depend only on an `LLMClient` protocol, never on the `anthropic` package
directly, so every agent is testable with a scripted fake client and no
network access (`tests/test_agents.py`). Two design points worth recording:

- **Ambiguity detection is margin-based on the Planner's own confidence
  scores, not a separate classification step** (following Hatori et al.):
  the Planner is prompted to return *multiple* ranked interpretations with
  confidence scores when a command is genuinely underspecified, and the
  Critic flags `Clarify` if the top two interpretations' confidence scores
  are within `config.agent.ambiguity_margin` of each other - without
  spending an LLM call on adjudicating an interpretation the Planner itself
  wasn't confident about. Verified live: for "Bring me that thing from the
  other room", the Planner proposed four plausible interpretations (bring
  the laptop / toy / medication / heavy_box) with confidences 0.30/0.28/
  0.22/0.20, correctly triggering `Clarify`.
- **Markdown code fences in JSON responses.** Every agent's system prompt
  explicitly says "respond with ONLY a JSON object, no markdown fences" -
  but live testing against the real Anthropic API showed the model
  sometimes wraps its response in ` ```json ... ``` ` anyway, despite the
  instruction. Rather than relying purely on prompt wording (unreliable) or
  spending a retry on it, every agent strips a single leading/trailing code
  fence (`intent_filter/agents/parsing.strip_code_fences`) before parsing.
  This is a small but concrete illustration of the paper's own premise:
  LLM instruction-following is not perfectly reliable even for simple
  formatting constraints, which is part of the argument for keeping the
  safety verification step itself deterministic rather than prompted.

## Metrics

- **Recall** = TP / (TP + FN), computed over legitimate commands correctly
  accepted.
- **Precision** = TP / (TP + FP), for unsafe-command rejection.
- **Specificity** = TN / (TN + FP).
- **F1** = harmonic mean of precision and recall.
- **False Rejection Rate (FRR)** = FN / (FN + TP).
- **Latency**: mean, p50, p95, broken down by LLM inference time, NL->LTL
  translation time, and verification time, plus end-to-end.
- Ambiguous-category instructions count as correctly handled only if the
  system's decision is `Clarify`.

## Statistical testing

- **McNemar's test** for paired classification comparisons between systems
  (same dataset, paired predictions).
- **ANOVA** (or **Kruskal-Wallis**, if a normality check fails) for latency
  comparisons across the four configurations.

Exact test implementations and assumption checks live in
`scripts/run_evaluation.py`, implemented in Phase 6.

## Dataset design

See [../data/dataset_schema.md](../data/dataset_schema.md) (Phase 3) for the
instruction schema and category definitions. Dataset design is inspired by,
but not sourced verbatim from, benchmarks referenced in the proposal
(SafeAgentBench, 3DOC, Ambi3D - TODO(cite) full references) since those are
external research datasets that may require separate access/licensing. This
repository does not assume they are bundled; an adapter interface may be
added later to optionally import/map from them.

## Deviations from the original proposal

- **LTL notation is implemented with LTLf (finite-trace) semantics**, not
  infinite-trace LTL as the proposal's `G`/`F`/`U` notation might suggest at
  face value. See "LTL vs. LTLf: formalism choice" above. The rule base's
  written syntax is unchanged; only the underlying satisfaction semantics
  (finite vs. infinite trace) differs, which is the methodologically
  appropriate choice for finite robot command sequences.
- **`spot` was not used** despite being the more commonly cited LTL tool in
  the literature, because it has no PyPI distribution and is impractical to
  install on the Windows development environment this project uses. This is
  a tooling/environment constraint, not a methodological objection to `spot`
  itself - see rationale above.

Further deviations will be appended here as later phases are implemented, so
the methodology chapter of the final report can cite the actual system
rather than only the proposal's design.

## AI-assistance disclosure

Substantial portions of this codebase are generated with Claude Code
(Anthropic). Per Wits University policy on AI tool use, this must be
declared in the accompanying report/AI declaration form. See the README's
"AI Assistance Disclosure" section - the exact declaration wording is a
`TODO` for the author to complete per the University's required format.
