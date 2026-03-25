#!/usr/bin/env bash
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

source ~/.bashrc
conda activate baseline_env

# First 4 original rows are intentionally disabled:
# "numeric_cn_simple numeric_de_simple numeric_es_simple"
# "numeric_cn_simple_ipa numeric_de_simple_ipa numeric_es_simple_ipa"
# "numeric_cn_simple_pos numeric_de_simple_pos numeric_es_simple_pos"
# "numeric_cn_simple_ipa_pos numeric_de_simple_ipa_pos numeric_es_simple_ipa_pos"

# Remaining rows split into one model per task.
MODELS=(
  "numeric_cn_base"
  "numeric_de_base"
  "numeric_es_base"
  "numeric_cn_base_ipa"
  "numeric_de_base_ipa"
  "numeric_es_base_ipa"
  "numeric_cn_base_pos"
  "numeric_de_base_pos"
  "numeric_es_base_pos"
  "numeric_cn_base_ipa_pos"
  "numeric_de_base_ipa_pos"
  "numeric_es_base_ipa_pos"
)

idx="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
if (( idx < 0 || idx >= ${#MODELS[@]} )); then
  echo "Invalid SLURM_ARRAY_TASK_ID=${idx}; expected 0..$(( ${#MODELS[@]} - 1 ))" >&2
  exit 2
fi

model="${MODELS[$idx]}"

echo "[$(date '+%F %T')] Starting task ${idx} with model: ${model}"
python run_pipeline.py \
  --finetune_num \
  --predict_num \
  --models_to_run "${model}" \
  --model_params_path models/num_model_parameters.csv

echo "[$(date '+%F %T')] Finished task ${idx}"
