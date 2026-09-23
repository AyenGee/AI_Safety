#!/bin/bash
# Runs cluster/pilot_timing.py on a single compute node - the FIRST thing to
# submit on the cluster, before any real experiment job. Its output
# (cluster/logs/pilot_timing_result.json) is what turns each experiment's
# target instruction count (docs/methodology.md's scale-up table) into a
# concrete shard count for cluster/run_experiment.slurm.
#
# `cluster/logs/` must exist before submitting: `mkdir -p cluster/logs`.
#
# Usage (from the repo root):
#   sbatch cluster/pilot.slurm

#SBATCH --job-name=intent-filter-pilot
#SBATCH --partition=batch
#SBATCH --cpus-per-task=6
#SBATCH --mem=20G
#SBATCH --time=06:00:00
#SBATCH --output=cluster/logs/%x_%j.out
#SBATCH --error=cluster/logs/%x_%j.err

set -euo pipefail

source ~/llm/env.sh
source .venv/bin/activate

ollama serve > "cluster/logs/ollama_pilot_${SLURM_JOB_ID}.log" 2>&1 &
OLLAMA_PID=$!
sleep 10

if ! ollama list | grep -q qwen3.5 || ! ollama list | grep -q gemma4; then
  echo "ERROR: qwen3.5/gemma4 not found via 'ollama list' (OLLAMA_HOST=$OLLAMA_HOST). Pull them first (see cluster/setup.md)." >&2
  exit 1
fi

python cluster/pilot_timing.py

kill "$OLLAMA_PID" 2>/dev/null || true
