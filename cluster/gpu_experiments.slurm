#!/bin/bash
# Runs the GPU timing pilot, then every scaled-up follow-on experiment from
# docs/methodology.md's scale-up table (everything except Phase 8, which has
# its own job: cluster/gpu_phase8.slurm), one after another on ONE biggpu
# node.
#
# Why sequential in one job: most experiment_*.py scripts only write their
# results JSON when they finish. Running them back to back means that if
# this job is ever cut off, only the experiment in progress is lost - every
# experiment already finished has its results/<name>_experiment.json on disk.
#
# Deliberately NOT included (per docs/methodology.md / project decision):
# planner_verifier, child_gate_misdirection, memory_stripped,
# memory_stripped_fully, new_misdirected, stove_temporal_inconsistency.
#
# Submit from the repo root:
#   mkdir -p cluster/logs
#   sbatch cluster/gpu_experiments.slurm
#
# To run only some experiments (e.g. to redo ones that failed), pass their
# names space-separated:
#   EXPERIMENTS="experiment_prompt_injection experiment_fact_injection" sbatch cluster/gpu_experiments.slurm
#
# No --gres line: this cluster's partitions report GRES (null); requesting
# the biggpu partition is what gets a GPU node.

#SBATCH --job-name=experiments-gpu
#SBATCH --partition=biggpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00
#SBATCH --output=cluster/logs/%x_%j.out
#SBATCH --error=cluster/logs/%x_%j.err

set -uo pipefail

DEFAULT_EXPERIMENTS="
experiment_instruction_decomposition
experiment_translator_accuracy
experiment_prompt_injection
experiment_authority_emergency_contradiction
experiment_grounded_critic
experiment_fact_injection
experiment_critic_model_swap
experiment_older_model
experiment_rules_removed
experiment_temporal_misdirection_large_scale
"
EXPERIMENTS="${EXPERIMENTS:-$DEFAULT_EXPERIMENTS}"

source ~/llm/env.sh
source .venv/bin/activate

# Same Ollama settings as cluster/gpu_phase8.slurm (see comments there).
export OLLAMA_CONTEXT_LENGTH=16384
export OLLAMA_KEEP_ALIVE=-1

echo "=== Experiments job on $(hostname), started $(date)"
echo "=== OLLAMA_HOST=$OLLAMA_HOST"
nvidia-smi || echo "WARNING: nvidia-smi failed - this node may have no usable GPU (inference would fall back to CPU)"

OLLAMA_LOG="cluster/logs/ollama_experiments_${SLURM_JOB_ID}.log"
ollama serve > "$OLLAMA_LOG" 2>&1 &
OLLAMA_PID=$!

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

# 1) Timing pilot on the GPU node (real seconds/call per model and role).
echo
echo "=== [pilot] started $(date)"
python cluster/pilot_timing.py --output "cluster/logs/pilot_timing_result_gpu_${SLURM_JOB_ID}.json" \
  > "cluster/logs/pilot_gpu_${SLURM_JOB_ID}.log" 2>&1
echo "=== [pilot] finished $(date), exit status $?"

# 2) Every experiment in turn. A failure in one does not stop the others.
SUMMARY=""
for EXP in $EXPERIMENTS; do
  echo
  echo "=== [$EXP] started $(date)"
  START=$(date +%s)
  python "scripts/${EXP}.py" > "cluster/logs/${EXP}_${SLURM_JOB_ID}.log" 2>&1
  STATUS=$?
  MINUTES=$(( ($(date +%s) - START) / 60 ))
  echo "=== [$EXP] finished $(date), exit status $STATUS, ${MINUTES} min"
  SUMMARY+="$(printf '%-50s status=%-3s %5s min' "$EXP" "$STATUS" "$MINUTES")"$'\n'
done

echo
echo "=== SUMMARY (status 0 = success; details in cluster/logs/<experiment>_${SLURM_JOB_ID}.log)"
printf '%s' "$SUMMARY"

echo "=== Prompt truncation warnings in Ollama log (should be 0):"
grep -ci "truncat" "$OLLAMA_LOG" || true

echo "=== All done $(date)"
kill "$OLLAMA_PID" 2>/dev/null || true
