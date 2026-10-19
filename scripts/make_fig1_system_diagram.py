#!/usr/bin/env python
"""Figure 1 for the report: the Multi-Agent + LTL intent-filtering pipeline.

Drawn from the implementation, not from the proposal's earlier figure:
intent_filter/systems/multi_agent_ltl.py (control flow), config/config.example.yaml
(model roles, ambiguity margin 0.15, 2 refinement attempts) and
config/safety_rules.yaml (10 LTLf rules). Two details the diagram deliberately
shows because they are easy to get wrong:

  * the Translator's formula is logged only and never gates a decision;
  * after a rule violation the Planner's revised plan goes straight back to
    the verifier - the Critic explains the violation but does not re-review
    the revised plan.

Writes fig00_system_diagram.{png,pdf,svg} (vector copies for the report).

Usage:
    python scripts/make_fig1_system_diagram.py [--out docs/open_weight_run/figures]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8985", "#e4e3df"
C_LLM, C_FORMAL = "#2a78d6", "#eb6834"  # same blue / orange as the results figures (no verifier / with verifier)
LLM_TINT, FORMAL_TINT = "#e8f0fb", "#fdece5"

plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE})


def box(ax, x0, y0, x1, y1, title, lines, color, tint, dashed=False, title_color="white", sub=None):
    w, h = x1 - x0, y1 - y0
    ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0,rounding_size=1.1",
                                fc=tint, ec="none", zorder=2))
    head = 5.2
    hc = color if not dashed else MUTED
    ax.add_patch(FancyBboxPatch((x0, y1 - head), w, head, boxstyle="round,pad=0,rounding_size=1.1",
                                fc=hc, ec="none", zorder=3))
    ax.add_patch(Rectangle((x0, y1 - head), w, head / 2, fc=hc, ec="none", zorder=3))
    ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0,rounding_size=1.1",
                                fc="none", ec=color, lw=1.8, ls="--" if dashed else "-", zorder=6))
    ax.text((x0 + x1) / 2, y1 - head / 2, title, ha="center", va="center", color=title_color,
            fontsize=12, fontweight="bold", zorder=4)
    ax.text((x0 + x1) / 2, (y0 + y1 - head) / 2 + (0.6 if sub else 0), "\n".join(lines), ha="center",
            va="center", color=INK, fontsize=10.2, linespacing=1.35, zorder=4)
    if sub:
        ax.text((x0 + x1) / 2, y0 + 1.5, sub, ha="center", va="center", color=INK2, fontsize=9.2,
                style="italic", zorder=4)


def arrow(ax, p0, p1, color=INK2, lw=1.8, ls="-", head=True, zorder=1):
    ax.annotate("", xy=p1, xytext=p0, zorder=zorder,
                arrowprops=dict(arrowstyle="-|>" if head else "-", lw=lw, color=color, ls=ls,
                                shrinkA=0, shrinkB=0, mutation_scale=15))


def poly(ax, pts, color=INK2, lw=1.8, ls="-", head=True):
    for a, b in zip(pts[:-2], pts[1:-1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=lw, ls=ls, solid_capstyle="butt", zorder=1)
    arrow(ax, pts[-2], pts[-1], color=color, lw=lw, ls=ls, head=head)


def label(ax, x, y, text, size=9.6, color=INK2, ha="center", weight="normal"):
    ax.text(x, y, text, ha=ha, va="center", fontsize=size, color=color, fontweight=weight, zorder=5,
            bbox=dict(boxstyle="round,pad=0.25", fc=SURFACE, ec="none"), linespacing=1.3)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="docs/open_weight_run/figures")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11.5, 6.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 58)
    ax.axis("off")

    MAIN_Y0, MAIN_Y1 = 22, 40
    # main row
    box(ax, 0.5, MAIN_Y0, 15.5, MAIN_Y1, "Input", ["Instruction (text)", "+ scene state:", "rooms, held objects,", "device states,", "issuing role"],
        INK2, "#efeeea")
    box(ax, 21, MAIN_Y0, 38.5, MAIN_Y1, "Planner (LLM)", ["Ranked interpretations", "with confidence scores", "and symbolic action", "sequences"],
        C_LLM, LLM_TINT, sub="gemma4:e4b")
    box(ax, 44.5, MAIN_Y0, 62, MAIN_Y1, "Critic (LLM)", ["1. Ambiguity check:", "top-2 confidence gap", "< 0.15 means Clarify", "2. Review top plan", "against the policy"],
        C_LLM, LLM_TINT, sub="qwen3.5:4b")
    box(ax, 68, MAIN_Y0, 85, MAIN_Y1, "LTLf verifier", ["Simulate the plan:", "get the state trajectory", "Check the trajectory", "against all 10 rules"],
        C_FORMAL, FORMAL_TINT, sub="deterministic, no LLM")
    box(ax, 90.5, MAIN_Y0 + 3, 99.5, MAIN_Y1 - 3, "Decision", ["Accept", "Reject", "Clarify"], INK, "#efeeea")

    # rule base above verifier
    box(ax, 68, 43.5, 85, 53.5, "Rule base", ["10 safety rules (LTLf),", "written by a human"], C_FORMAL, FORMAL_TINT)
    arrow(ax, (76.5, 43.5), (76.5, MAIN_Y1), color=C_FORMAL)

    # main flow arrows
    arrow(ax, (15.5, 31), (21, 31))
    arrow(ax, (38.5, 31), (44.5, 31))
    arrow(ax, (62, 31), (68, 31))
    arrow(ax, (85, 31), (90.5, 31))
    label(ax, 65, 33.6, "plan\naccepted", size=9)
    label(ax, 87.75, 33.2, "SAT", size=9, color=C_FORMAL, weight="bold")

    # critic early exit (Clarify / Reject) over the top to Decision
    poly(ax, [(53.25, MAIN_Y1), (53.25, 56), (95, 56), (95, MAIN_Y1 - 3)], color=C_LLM)
    label(ax, 61, 56, "Clarify (ambiguous) or Reject (policy violation)", size=9.6, color=C_LLM)

    # feedback loop from verifier back to planner
    poly(ax, [(76.5, MAIN_Y0), (76.5, 19.2), (29.75, 19.2), (29.75, MAIN_Y0)], color=C_FORMAL)
    label(ax, 54.4, 15.3, "Rule violated (UNSAT): the Critic explains the violation and the Planner\nrevises the plan, up to 2 times; the revised plan goes straight back to the verifier.\nStill UNSAT, or physically impossible plan: Reject.",
          size=9.6, color=C_FORMAL)

    # translator (logged only)
    poly(ax, [(8, MAIN_Y0), (8, 5.7), (21, 5.7)], color=MUTED, ls="--")
    box(ax, 21, 0.4, 38.5, 11.0, "Translator (LLM)", ["Instruction to", "LTL formula"], MUTED, "#f3f2ee", dashed=True, title_color="white")
    arrow(ax, (38.5, 5.7), (44.5, 5.7), color=MUTED, ls="--")
    box(ax, 44.5, 0.4, 62, 11.0, "Logged only", ["Never gates a decision;", "used to measure", "translation accuracy"], MUTED, "#f3f2ee", dashed=True)

    # legend
    for i, (c, t) in enumerate([(C_LLM, "LLM component (probabilistic)"), (C_FORMAL, "Formal component (deterministic)"),
                                (MUTED, "Dashed: logged, not used in decisions")]):
        y = 56.2 - i * 2.6
        ax.add_patch(Rectangle((0.5, y - 0.8), 2.2, 1.6, fc=c if i < 2 else "#f3f2ee", ec=c, lw=1.4,
                               ls="-" if i < 2 else "--", zorder=2))
        ax.text(3.6, y, t, va="center", ha="left", fontsize=9.6, color=INK)

    ax.text(50, 58.2, "", fontsize=1)
    fig.tight_layout(pad=0.4)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out / f"fig00_system_diagram.{ext}", dpi=220, bbox_inches="tight")
        print("wrote", out / f"fig00_system_diagram.{ext}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
