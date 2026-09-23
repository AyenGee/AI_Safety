#!/bin/bash
# Generic Slurm job-array template: fans one experiment's dataset out across
# several simultaneous batch nodes (see docs/methodology.md "Parallelizing
# across the cluster" and intent_filter/sharding.py). Each array task runs
# its own private Ollama server and processes one round-robin shard of the
# instructions, via scripts/run_evaluation.py's or an experiment_*.py
# script's --shard-index/--shard-count flags.
#
# `cluster/logs/` must exist before submitting - Slurm opens --output/--error
# before running anything in this script: `mkdir -p cluster/logs`.
#
# Usage (submit from the repo root - relative paths below depend on it):
#   SCRIPT=scripts/run_evaluation.py SHARD_COUNT=20 \
#     sbatch --array=0-19 cluster/run_experiment.slurm
#
#   SCRIPT=scripts/experiment_prompt_injection.py SHARD_COUNT=10 \
#     EXTRA_ARGS="--repeats 1" sbatch --array=0-9 cluster/run_experiment.slurm
#
# SHARD_COUNT must match --array's size (0-19 => 20 shards => SHARD_COUNT=20).
# Slurm exports the submitting shell's environment by default, so SCRIPT/
# SHARD_COUNT/EXTRA_ARGS set on the sbatch command line reach every task.
#
# A task killed by TIMEOUT (1-day cap on `batch`) can simply be resubmitted
# with the same --array range restricted to the missing task ids and
# --resume added to EXTRA_ARGS pointed at that task's own run dir - each
# shard's run dir is independent (see run_evaluation.py's shard-suffixed
# naming), so this never touches the other shards' output.

#SBATCH --job-name=intent-filter
#SBATCH --partition=batch
#SBATCH --cpus-per-task=6
#SBATCH --mem=20G
#SBATCH --time=23:00:00
#SBATCH --output=cluster/logs/%x_%A_%a.out
#SBATCH --error=cluster/logs/%x_%A_%a.err

set -euo pipefail

: "${SCRIPT:?Set SCRIPT=scripts/run_evaluation.py (or an experiment_*.py) before sbatch}"
: "${SHARD_COUNT:?Set SHARD_COUNT to match --array's size, e.g. --array=0-19 => SHARD_COUNT=20}"

source ~/llm/env.sh
source .venv/bin/activate

ollama serve > "cluster/logs/ollama_${SLURM_JOB_ID}.log" 2>&1 &
OLLAMA_PID=$!
sleep 10

# Fail fast if either model isn't present on this node's Ollama install,
# rather than burning the time budget on every instruction before erroring.
if ! ollama list | grep -q qwen3.5 || ! ollama list | grep -q gemma4; then
  echo "ERROR: qwen3.5/gemma4 not found via 'ollama list' (OLLAMA_HOST=$OLLAMA_HOST). Pull them first (see cluster/setup.md)." >&2
  exit 1
fi

echo "Shard $SLURM_ARRAY_TASK_ID/$SHARD_COUNT on $(hostname), OLLAMA_HOST=$OLLAMA_HOST"

python "$SCRIPT" --shard-index "$SLURM_ARRAY_TASK_ID" --shard-count "$SHARD_COUNT" ${EXTRA_ARGS:-}

kill "$OLLAMA_PID" 2>/dev/null || true
