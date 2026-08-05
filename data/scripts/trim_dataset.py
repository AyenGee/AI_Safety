"""One-off Phase 7 script: deterministically trim data/instructions.jsonl from
300 down to 200 rows (per the researcher's supervisor's cost guidance),
while preserving every safety rule's violating+safe example pair and an
even spread of phrasing diversity within each rule/category group.

Not part of the ongoing regeneration workflow (see dataset_schema.md) - this
is a one-time rebalancing step, kept in the repo for the audit trail of how
the 200-example dataset was derived from the 300-example one.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATASET_PATH = REPO_ROOT / "data" / "instructions.jsonl"

TARGET_TOTAL = {
    "legitimate": 50,
    "unsafe": 57,
    "misdirected": 33,
    "ambiguous": 60,
}


def even_stride_select(items: list, target: int) -> list:
    """Deterministically pick `target` items evenly spread across `items`,
    preserving original relative order - avoids clustering the kept rows in
    just one region (e.g. only the earliest-written, most similar phrasings).
    """
    n = len(items)
    if target >= n:
        return list(items)
    if target <= 0:
        return []
    indices = sorted({round(i * (n - 1) / (target - 1)) for i in range(target)}) if target > 1 else [0]
    # Rounding to a set can collapse to fewer than `target` unique indices;
    # top up with the next unused index (in order) until we hit target.
    indices = list(indices)
    used = set(indices)
    i = 0
    while len(indices) < target and i < n:
        if i not in used:
            indices.append(i)
            used.add(i)
        i += 1
    indices.sort()
    return [items[i] for i in indices[:target]]


def largest_remainder_allocation(group_sizes: dict, total_target: int, min_per_group: int = 2) -> dict:
    """Proportionally allocate `total_target` across groups sized by `group_sizes`,
    guaranteeing each group gets at least `min_per_group`, summing exactly to `total_target`.
    """
    groups = list(group_sizes)
    total_current = sum(group_sizes.values())
    raw = {g: group_sizes[g] / total_current * total_target for g in groups}
    floor_alloc = {g: max(min_per_group, int(raw[g])) for g in groups}

    # Adjust so the sum matches exactly, biasing by largest fractional remainder.
    diff = total_target - sum(floor_alloc.values())
    remainders = sorted(groups, key=lambda g: raw[g] - int(raw[g]), reverse=True)
    idx = 0
    while diff > 0:
        g = remainders[idx % len(remainders)]
        floor_alloc[g] += 1
        diff -= 1
        idx += 1
    while diff < 0:
        g = remainders[(-idx - 1) % len(remainders)]
        if floor_alloc[g] > min_per_group:
            floor_alloc[g] -= 1
            diff += 1
        idx += 1
    return floor_alloc


def main() -> None:
    rows = []
    with open(DATASET_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)

    kept: list[dict] = []

    # --- legitimate: always keep the 8 rule-tagged safe-counterpart rows,
    # evenly subsample the generic filler rows for the remainder.
    legit = by_cat["legitimate"]
    rule_tagged = [r for r in legit if r.get("related_rule_ids")]
    filler = [r for r in legit if not r.get("related_rule_ids")]
    filler_target = TARGET_TOTAL["legitimate"] - len(rule_tagged)
    kept += rule_tagged
    kept += even_stride_select(filler, filler_target)

    # --- unsafe / misdirected: group by exact related_rule_ids tuple (each
    # group corresponds to one rule-violation scenario family), allocate the
    # category target proportionally across groups, then evenly subsample
    # within each group.
    for category in ("unsafe", "misdirected"):
        rows_in_cat = by_cat[category]
        groups: dict[tuple, list[dict]] = defaultdict(list)
        for r in rows_in_cat:
            groups[tuple(r.get("related_rule_ids", []))].append(r)

        group_sizes = {k: len(v) for k, v in groups.items()}
        allocation = largest_remainder_allocation(group_sizes, TARGET_TOTAL[category], min_per_group=2)

        for key, group_rows in groups.items():
            kept += even_stride_select(group_rows, allocation[key])

    # --- ambiguous: no rule ties, evenly subsample the whole category.
    kept += even_stride_select(by_cat["ambiguous"], TARGET_TOTAL["ambiguous"])

    # Preserve original file order among kept rows.
    kept_ids = {r["id"] for r in kept}
    ordered_kept = [r for r in rows if r["id"] in kept_ids]

    with open(DATASET_PATH, "w", encoding="utf-8", newline="\n") as f:
        for r in ordered_kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Trimmed {len(rows)} -> {len(ordered_kept)} rows")
    counts = defaultdict(int)
    for r in ordered_kept:
        counts[r["category"]] += 1
    print(dict(counts))


if __name__ == "__main__":
    main()
