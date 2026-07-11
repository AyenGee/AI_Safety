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

**Decision (to be finalized in Phase 2):** evaluate `flloat` (pure-Python
LTLf, pip-installable) against `spot` (canonical LTL-to-Buchi-automaton
toolkit, but Linux/conda-oriented with weak native Windows support - the
development machine for this repository is Windows) and `ltlf2dfa`. Current
expectation is to adopt LTLf via `flloat` for both practical (cross-platform,
pure Python) and methodological (finite-horizon domain fit) reasons, while
keeping the safety-rule YAML syntax visually close to standard LTL notation
(`G`, `F`, `U`, `X`) since the finite-trace and infinite-trace operators
share the same syntax and differ only in semantics over finite vs. infinite
traces. This section will be updated with the final decision and rationale
once Phase 2 is implemented and benchmarked.

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

None yet (Phase 1: environment/config scaffolding only). This section will
be updated as implementation choices are made in later phases, so the
methodology chapter of the final report can cite the actual system rather
than only the proposal's design.

## AI-assistance disclosure

Substantial portions of this codebase are generated with Claude Code
(Anthropic). Per Wits University policy on AI tool use, this must be
declared in the accompanying report/AI declaration form. See the README's
"AI Assistance Disclosure" section - the exact declaration wording is a
`TODO` for the author to complete per the University's required format.
