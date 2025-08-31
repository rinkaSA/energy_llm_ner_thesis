#!/bin/bash
#SBATCH --job-name=vllm_client
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_out/slurm_eval_output_logs/ev_%j.out
#SBATCH --error=slurm_out/slurm_eval_output_logs/ev_%j.err

WORKSPACE_DIR="/data/horse/ws/irve354e-energy_llm_ner/energy_ner_llm/src/serve_vllm"
VENV_PATH="/data/horse/ws/irve354e-energy_llm_ner/super_weights/universal-ner-py311"
METRICS_DIR="${WORKSPACE_DIR}/gpu_metrics_server"
METRICS_SCRIPT="${WORKSPACE_DIR}/collect_gpu_nvml.py"   

language="de"

if [[ -f "${WORKSPACE_DIR}/.env" ]]; then
  set -a
  source "${WORKSPACE_DIR}/.env"
  set +a
fi


if [[ -z "${ENERGY_URL:-}" ]]; then
  echo "[FATAL] ENERGY_URL is not set (export it or put it in ${WORKSPACE_DIR}/.env)"
  exit 2
fi

METRICS_CSV="${METRICS_DIR}/metrics_col_xt_${language}_${SLURM_JOB_ID}.csv"
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
nohup python3 "${METRICS_SCRIPT}" > "${METRICS_DIR}/metrics_col_xt_${language}.log" 2>&1 &
SCRAPER_PID=$!
echo "[INFO] scraper PID=${SCRAPER_PID}"

trap 'echo "[INFO] stopping scraper ${SCRAPER_PID}"; kill ${SCRAPER_PID} 2>/dev/null || true' EXIT

cd "${WORKSPACE_DIR}" || exit 1
srun bash -lc "source ${VENV_PATH}/bin/activate && python3 evaluation_scripts/xtreme/lang_eval_mlflow_mi_gol.py --language ${language} --model /gollie"
