#!/bin/bash
# Phase 8 main evaluation (scripts/run_evaluation.py: 450 examples x 3
# repeats x 4 systems + 3 ablations) on the biggpu partition, split into 2
# halves that run as a 2-task job array (one GPU node each). Each half is
# estimated at well under biggpu's 3-day cap; run_evaluation.py flushes every
# completed run to raw_results.jsonl immediately, so a half that is cut off
# can be resumed (see "Resuming" below).
#
# Submit from the repo root (relative paths below depend on it):
#   mkdir -p cluster/logs
#   sbatch cluster/gpu_phase8.slurm
#
# When BOTH halves have finished, merge them on the login node:
#   source .venv/bin/activate
#   python scripts/merge_shards.py --shard-dirs "results/2026*_shard*of2" --output results/phase8_gpu_merged
#
# Resuming a half that was cut off (e.g. half 1, whose run dir was printed
# near the top of its .out file):
#   EXTRA_ARGS="--resume results/<that_run_dir>" sbatch --array=1 cluster/gpu_phase8.slurm
#
# No --gres line: this cluster's partitions report GRES (null); requesting
# the biggpu partition is what gets a GPU node.

#SBATCH --job-name=phase8-gpu
#SBATCH --partition=biggpu
#SBATCH --array=0-1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00
#SBATCH --output=cluster/logs/%x_%A_%a.out
#SBATCH --error=cluster/logs/%x_%A_%a.err

set -uo pipefail

SHARD_COUNT=2

source ~/llm/env.sh
source .venv/bin/activate

# Ollama server settings for this job:
# - CONTEXT_LENGTH: guarantees the full system prompt (ontology + safety
#   rules + instruction) fits the model's context window, instead of relying
#   on Ollama's default, which silently truncates longer prompts.
# - KEEP_ALIVE=-1: keep both models loaded for the whole job rather than
#   unloading/reloading them between Planner/Critic/Translator calls.
export OLLAMA_CONTEXT_LENGTH=16384
export OLLAMA_KEEP_ALIVE=-1

echo "=== Phase 8 half $SLURM_ARRAY_TASK_ID/$SHARD_COUNT on $(hostname), started $(date)"
echo "=== OLLAMA_HOST=$OLLAMA_HOST"
nvidia-smi || echo "WARNING: nvidia-smi failed - this node may have no usable GPU (inference would fall back to CPU)"

OLLAMA_LOG="cluster/logs/ollama_phase8_${SLURM_JOB_ID}.log"
ollama serve > "$OLLAMA_LOG" 2>&1 &
OLLAMA_PID=$!

# Wait (up to ~2 minutes) for the server to answer, instead of a fixed sleep.
for _ in $(seq 1 60); do
  ollama list > /dev/null 2>&1 && break
  sleep 2
done

if ! ollama list | grep -q qwen3.5 || ! ollama list | grep -q gemma4; then
  echo "ERROR: qwen3.5/gemma4 not found via 'ollama list'. Pull them first (see cluster/setup.md)." >&2
  kill "$OLLAMA_PID" 2>/dev/null
  exit 1
fi

echo "=== Ollama GPU detection:"
grep -iE "inference compute|no compatible GPUs" "$OLLAMA_LOG" || echo "(no GPU detection line found yet - check $OLLAMA_LOG)"

python scripts/run_evaluation.py --shard-index "$SLURM_ARRAY_TASK_ID" --shard-count "$SHARD_COUNT" ${EXTRA_ARGS:-}
STATUS=$?

echo "=== Prompt truncation warnings in Ollama log (should be 0):"
grep -ci "truncat" "$OLLAMA_LOG" || true

echo "=== Finished $(date) with exit status $STATUS"
kill "$OLLAMA_PID" 2>/dev/null || true
exit $STATUS
