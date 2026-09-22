# Running this project's evaluation harness on mscluster

Assumes `mscluster_LLM_setup_guide.pdf`'s Parts 1-6 are already done (Ollama
installed under `~/llm`, `qwen3.5:4b` and `gemma4:e4b` pulled, `~/llm/env.sh`
in place). This document is the project-specific layer on top of that: how
to get this repo running there and start real experiment jobs.

## One-time setup (on the login node)

```bash
cd ~
git clone <this repo's URL> intent-filter   # or: scp -r the repo from your machine
cd intent-filter
python3 -m venv .venv
source .venv/bin/activate
pip install .              # pyyaml, pydantic, flloat, scipy, statsmodels, matplotlib -
                            # no anthropic package needed, this project talks to Ollama now
mkdir -p cluster/logs       # Slurm needs this to exist before #SBATCH --output/--error can be opened
```

No API key, no `.env` file needed - `OllamaLLMClient` talks to the local
Ollama server via `OLLAMA_HOST`, which `~/llm/env.sh` already exports per
Slurm job.

## Step 1: calibration pilot (always run this first)

```bash
sbatch cluster/pilot.slurm
squeue -u $USER                          # wait for it to finish (~30 min cap)
cat cluster/logs/intent-filter-pilot_*.out
cat cluster/logs/pilot_timing_result.json
```

This measures real seconds/call for each (model, agent role) pair on this
cluster's actual 6-core nodes - report the summary table back so the
per-experiment shard counts below can be set from real numbers, not a guess.

## Step 2: run a real experiment, sharded across the cluster

Each experiment script accepts `--shard-index`/`--shard-count` (see
`intent_filter/sharding.py`). One `sbatch --array=...` submits every shard
of one experiment at once; Slurm fans them out across however many `batch`
nodes are free.

```bash
mkdir -p cluster/logs
# Phase 8 main dataset, split 20 ways:
SCRIPT=scripts/run_evaluation.py SHARD_COUNT=20 \
  sbatch --array=0-19 cluster/run_experiment.slurm

# A smaller experiment script, split 10 ways, with extra flags passed through:
SCRIPT=scripts/experiment_prompt_injection.py SHARD_COUNT=10 \
  EXTRA_ARGS="--repeats 1" sbatch --array=0-9 cluster/run_experiment.slurm
```

`SHARD_COUNT` must equal the `--array` size (`0-19` is 20 tasks). Actual
shard counts per experiment are set once Step 1's numbers are in - see
`docs/methodology.md`'s scale-up table for target instruction counts.

## Step 3: monitor

```bash
squeue -u $USER                 # R = running, PD = waiting
tail -f cluster/logs/intent-filter_<jobid>_<taskid>.out
```

A shard killed by `TIMEOUT` (1-day cap on `batch`) can be resubmitted for
just that task id with `--resume` added to `EXTRA_ARGS`, pointed at that
shard's own run dir (each shard's dir is independent - see
`run_evaluation.py`'s `_shard<i>of<n>` naming).

## Step 4: merge shards into one report

Once every shard for an experiment has finished:

```bash
python scripts/merge_shards.py --shard-dirs "results/2026*_shard*of20" --output results/<name>_merged
```

This concatenates every shard's `raw_results.jsonl` and produces one full
report (metrics/stats/unsafety-breakdown/plots) over the complete dataset -
see `scripts/merge_shards.py`'s docstring for why this can't be done
per-shard.

## Step 5: bring results home

```bash
# From your own machine:
scp -r eazubuike@<cluster-address>:~/intent-filter/results/<name>_merged ./results/
```

## Ground rules (repeating the cluster guide's, since they still apply)

- Never run anything heavy on the login node - `sbatch`/`srun` only for
  actual model inference.
- Never touch `~/base_code` or `~/semantic-dementia-clip-main` - unrelated
  to this project.
- Start small: run the pilot, then one small experiment shard, before
  submitting a 100-shard Phase 8 job array.
