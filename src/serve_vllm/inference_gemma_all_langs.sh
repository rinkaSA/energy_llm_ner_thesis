#!/bin/bash
#SBATCH --job-name=gemma_ner_all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=ev_gemma_all_langs_%j.out
#SBATCH --error=ev_gemma_all_langs_%j.err

WORKSPACE_DIR="/data/horse/ws/irve354e-energy_llm_ner/energy_ner_llm/src/serve_vllm"
VENV_PATH="/data/horse/ws/irve354e-energy_llm_ner/super_weights/universal-ner-py311"
METRICS_DIR="${WORKSPACE_DIR}/gpu_metrics_server"
METRICS_SCRIPT="${WORKSPACE_DIR}/collect_gpu_nvml.py"   

# All languages in XTREME
languages=("en" "de" "ar" "bg" "zh")
prompt_style=5  # Using prompt style 5 for all languages
batch_size=128
max_new_tokens=60
sample_limit=10000
cooldown_seconds=30  # Longer cooldown between languages
use_generic_template=true  # Using the generic template for all languages

if [[ -f "${WORKSPACE_DIR}/.env" ]]; then
  set -a
  source "${WORKSPACE_DIR}/.env"
  set +a
fi

if [[ -z "${ENERGY_URL:-}" ]]; then
  echo "[FATAL] ENERGY_URL is not set (export it or put it in ${WORKSPACE_DIR}/.env)"
  exit 2
fi

METRICS_CSV="${METRICS_DIR}/metrics_col_xt_all_langs_${SLURM_JOB_ID}.csv"
METRICS_INTERVAL_MS=100

export METRICS_CSV
export METRICS_INTERVAL_MS

export JOB_ID=${SLURM_JOB_ID}
echo "Job ID is $JOB_ID"

echo "[INFO] ENERGY_URL=${ENERGY_URL}"
echo "[INFO] METRICS_CSV=${METRICS_CSV}"
echo "[INFO] METRICS_INTERVAL_MS=${METRICS_INTERVAL_MS}"

source "${VENV_PATH}/bin/activate"

echo "[INFO] starting GPU metrics scraper..."
nohup python3 "${METRICS_SCRIPT}" > "${METRICS_DIR}/metrics_col_xt_all_langs.log" 2>&1 &
SCRAPER_PID=$!
echo "[INFO] scraper PID=${SCRAPER_PID}"

trap 'echo "[INFO] stopping scraper ${SCRAPER_PID}"; kill ${SCRAPER_PID} 2>/dev/null || true' EXIT

for lang in "${languages[@]}"; do
  # record the start timestamp for this run
  start_timestamp=$(date -u +"%Y-%m-%dT%H:%M:%S.%6N")
  
  template_flag=""
  if [[ "${use_generic_template}" == "true" ]]; then
    template_flag="--use-generic-template"
    echo "[INFO] Starting run for language=${lang} with system_prompt_choice=${prompt_style} using generic template at ${start_timestamp}"
  else
    echo "[INFO] Starting run for language=${lang} with system_prompt_choice=${prompt_style} at ${start_timestamp}"
  fi
  
  srun bash -lc "
    set -euo pipefail

    echo 'Activating venv: ${VENV_PATH}'
    source ${VENV_PATH}/bin/activate

    cd ${WORKSPACE_DIR}

    # Using chat API (process_batch_chat) for Gemma
    python ${WORKSPACE_DIR}/evaluation_scripts/xtreme/lang_eval_mlflow_mi_gol.py \
      --language ${lang} \
      --model /gemma-3-4b-it \
      --system-prompt-choice ${prompt_style} \
      --batch-size ${batch_size} \
      --max-new-tokens ${max_new_tokens} \
      --limit ${sample_limit} \
      --run-start-timestamp \"${start_timestamp}\" \
      ${template_flag}
  "
  
  echo "[INFO] Completed run for language=${lang}"
  
  # wait between runs to allow GPU utilization to drop
  echo "[INFO] Cooling down for ${cooldown_seconds} seconds before next language"
  sleep ${cooldown_seconds}
done

echo "[INFO] All language evaluations completed"
