#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

mkdir -p logs

# Run 12 single-model tasks sequentially (%1) as Slurm array tasks.
# Pass extra sbatch options as args, e.g.:
#   ./run_experiments.sh --partition=gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G
sbatch \
  --job-name=num-exp \
  --time=08:00:00 \
  --array=0-11%1 \
  --output=logs/%x_%A_%a.out \
  --error=logs/%x_%A_%a.err \
  "$@" \
  "$script_dir/run_experiment_task.sh"
