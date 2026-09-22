"""Splits a flat list of items across N parallel workers - used to spread one
experiment's instructions across several simultaneous Slurm jobs (job array,
see cluster/run_experiment.slurm) so CPU-only inference on the cluster
finishes in a reasonable number of wall-clock hours instead of running one
instruction at a time on a single node.

Round-robin (`items[shard_index::shard_count]`), not contiguous blocks:
every dataset in this project is stored grouped by id prefix
(legit_/unsafe_/misd_/amb_, or similarly for the smaller experiment case
lists), so a contiguous slice would hand entire shards nothing but one
category - round-robin keeps every shard's category mix representative of
the whole dataset without needing separate stratification logic.
"""

from __future__ import annotations

from typing import Sequence, TypeVar

T = TypeVar("T")


def shard_slice(items: Sequence[T], shard_index: int, shard_count: int) -> list[T]:
    if shard_count < 1:
        raise ValueError(f"shard_count must be >= 1, got {shard_count}")
    if not (0 <= shard_index < shard_count):
        raise ValueError(f"shard_index must be in [0, {shard_count}), got {shard_index}")
    return list(items[shard_index::shard_count])
