#!/usr/bin/env python
"""Manual/debug CLI: run one natural-language instruction through any of the
four intent-filtering systems and print the decision plus the full stage
trace (agent outputs, verifier outcomes, latencies).

Usage:
    python scripts/run_single_instruction.py --system multi_agent_ltl \
        --text "Bring the knife to the child's room"

    python scripts/run_single_instruction.py --system single_llm \
        --text "Get me my medication" --role child
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from intent_filter.agents.client import AnthropicLLMClient  # noqa: E402
from intent_filter.config import load_config, load_secrets  # noqa: E402
from intent_filter.decision import SystemContext  # noqa: E402
from intent_filter.environment import initial_state, load_ontology, load_safety_rules  # noqa: E402
from intent_filter.systems import SYSTEMS  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--system",
        required=True,
        choices=sorted(SYSTEMS),
        help="Which pipeline system to run the instruction through.",
    )
    parser.add_argument("--text", required=True, help="The natural-language instruction.")
    parser.add_argument(
        "--role", default="owner", help="Role issuing the instruction (owner/child/guest). Default: owner."
    )
    parser.add_argument(
        "--room", default=None, help="Override the agent's starting room. Default: environment default."
    )
    parser.add_argument(
        "--config", default=None, help="Path to config.yaml. Default: config/config.yaml, falling back to config.example.yaml."
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()

    config = load_config(args.config)
    secrets = load_secrets()
    ontology = load_ontology(config.environment.ontology_path)
    rule_base = load_safety_rules(config.environment.safety_rules_path)
    client = AnthropicLLMClient(api_key=secrets.anthropic_api_key)

    state = initial_state(ontology, issuing_role=args.role)
    if args.room:
        state = state.with_updates(agent_room=args.room)

    ctx = SystemContext(
        client=client,
        models=config.models,
        ontology=ontology,
        rule_base=rule_base,
        ambiguity_margin=config.agent.ambiguity_margin,
        max_refinement_attempts=config.agent.max_refinement_attempts,
        translation_max_retries=config.agent.translation_max_retries,
    )

    run_fn = SYSTEMS[args.system]
    result = run_fn(args.text, state, ctx)

    print(f"System:    {args.system}")
    print(f"Command:   {args.text!r}")
    print(f"Decision:  {result.decision}")
    print(f"Rationale: {result.rationale}")
    print(f"Latency:   {result.total_latency_seconds:.2f}s total, refinement_attempts={result.refinement_attempts}")
    print()
    print("Stage trace:")
    for i, stage in enumerate(result.stages):
        print(f"  [{i}] {stage.stage} ({stage.latency_seconds:.2f}s)")
        print(f"      {json.dumps(stage.detail, indent=2, default=str)}".replace("\n", "\n      "))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
