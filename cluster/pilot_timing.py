#!/usr/bin/env python
"""Calibration pilot: measures real seconds/call for each (model, agent
role) pair against the cluster's own Ollama server, using the SAME prompt-
building functions the real pipeline uses (agents/planner.py's plan(),
critic.py's review(), translator.py's translate(), single_llm.py's run()) -
not toy prompts - so the numbers reflect actual production latency.

Why this has to run before any of the scaled-up experiments: CPU-only
inference on a 6-core batch node has no reliable published tokens/sec figure
for qwen3.5:4b/gemma4:e4b under this project's specific prompt lengths, and
guessing a number to size 9 experiments' shard counts around would repeat a
mistake already corrected once in this project (drawing a conclusion, then
checking it against evidence, instead of the other way around - see
docs/methodology.md's H1 discussion). This script produces the real number.

Run on a compute node (via cluster/pilot.slurm), never on the login node -
same rule as everything else in mscluster_LLM_setup_guide.pdf.

Usage:
    python cluster/pilot_timing.py [--n-examples 5]

Writes cluster/logs/pilot_timing_result.json with mean/median/p95 seconds
per (model, role), plus a rough total-experiment-wall-clock estimate for
each target-N in docs/methodology.md's scale-up table.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from intent_filter.agents.client import OllamaLLMClient  # noqa: E402
from intent_filter.agents.critic import review  # noqa: E402
from intent_filter.agents.planner import plan  # noqa: E402
from intent_filter.agents.single_llm import run as run_single_llm  # noqa: E402
from intent_filter.agents.translator import translate  # noqa: E402
from intent_filter.config import load_config  # noqa: E402
from intent_filter.dataset import load_dataset  # noqa: E402
from intent_filter.environment import load_ontology, load_safety_rules  # noqa: E402

MODELS = ["qwen3.5:4b", "gemma4:e4b"]

# Rows to base this pilot on: one non-trivial example per core category, so
# both a short and a long realistic prompt get timed - a single fixed prompt
# would risk under- or over-estimating typical latency in this project's
# actual prompt-length range.
SAMPLE_IDS = ["legit_001", "unsafe_001", "misd_001", "amb_001"]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-examples", type=int, default=None, help="Override the number of sample instructions timed per (model, role). Default: all of SAMPLE_IDS.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--output", default="cluster/logs/pilot_timing_result.json")
    return parser


def _time_call(fn, *args, **kwargs) -> float | None:
    start = time.monotonic()
    try:
        fn(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 - a pilot failure is data, not a crash
        print(f"    call failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return None
    return time.monotonic() - start


def main() -> int:
    args = build_arg_parser().parse_args()
    config = load_config(args.config)
    ontology = load_ontology(config.environment.ontology_path)
    rule_base = load_safety_rules(config.environment.safety_rules_path)
    examples_by_id = {e.id: e for e in load_dataset(config.dataset.path)}

    sample_ids = SAMPLE_IDS[: args.n_examples] if args.n_examples else SAMPLE_IDS
    examples = [examples_by_id[i] for i in sample_ids if i in examples_by_id]
    if not examples:
        print("None of SAMPLE_IDS were found in the current dataset - check data/instructions.jsonl.", file=sys.stderr)
        return 1

    client = OllamaLLMClient(base_url=config.ollama.base_url, timeout=config.ollama.timeout, max_retries=config.ollama.max_retries)

    timings: dict[str, dict[str, list[float]]] = {m: {"planner": [], "critic": [], "translator": [], "single_llm": []} for m in MODELS}

    for model in MODELS:
        print(f"=== {model} ===")
        for example in examples:
            state = example.scene_context.to_world_state(ontology)
            print(f"  [{example.id}] planner...")
            planner_output = None
            t = _time_call(plan, client, model, example.instruction_text, state, ontology)
            if t is not None:
                timings[model]["planner"].append(t)
                try:
                    planner_output = plan(client, model, example.instruction_text, state, ontology)
                except Exception:  # noqa: BLE001
                    planner_output = None

            if planner_output is not None:
                print(f"  [{example.id}] critic...")
                t = _time_call(
                    review, client, model, example.instruction_text, planner_output, state, ontology,
                    rule_base, config.agent.ambiguity_margin,
                )
                if t is not None:
                    timings[model]["critic"].append(t)

            print(f"  [{example.id}] translator...")
            t = _time_call(translate, client, model, example.instruction_text, ontology)
            if t is not None:
                timings[model]["translator"].append(t)

            print(f"  [{example.id}] single_llm...")
            t = _time_call(run_single_llm, client, model, example.instruction_text, state, ontology, rule_base)
            if t is not None:
                timings[model]["single_llm"].append(t)

    summary = {}
    for model, by_role in timings.items():
        summary[model] = {}
        for role, values in by_role.items():
            if not values:
                summary[model][role] = None
                continue
            summary[model][role] = {
                "n": len(values),
                "mean_s": round(statistics.mean(values), 2),
                "median_s": round(statistics.median(values), 2),
                "max_s": round(max(values), 2),
                "min_s": round(min(values), 2),
            }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"raw_timings_s": timings, "summary": summary}, f, indent=2)

    print(f"\n=== Summary (mean seconds/call) ===")
    print(f"{'Model':<14}{'Planner':>10}{'Critic':>10}{'Translator':>12}{'Single-LLM':>12}")
    for model, by_role in summary.items():
        row = f"{model:<14}"
        for role in ("planner", "critic", "translator", "single_llm"):
            s = by_role.get(role)
            row += f"{s['mean_s'] if s else 'n/a':>10}" if role != "translator" else f"{s['mean_s'] if s else 'n/a':>12}"
        print(row)

    print(f"\nFull results written to {output_path}")
    print(
        "\nNext: use these numbers to convert each experiment's target-N (docs/methodology.md's "
        "scale-up table) into a shard count that fits comfortably inside one `batch` node-day - "
        "see cluster/setup.md."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
