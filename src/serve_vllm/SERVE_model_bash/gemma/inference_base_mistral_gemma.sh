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


set -a
source "${WORKSPACE_DIR}/.env"
set +a

# Extract the server NODE_NAME from the .env file instead of using the current node
SERVER_NODE_NAME=$(grep "^NODE_NAME=" "${WORKSPACE_DIR}/.env" | cut -d'"' -f2)
export COMPLETIONS_URL="http://${SERVER_NODE_NAME}:8000/v1/completions"
echo "Using server node: ${SERVER_NODE_NAME}"
echo "Using COMPLETIONS_URL: ${COMPLETIONS_URL}"

srun bash -c "
source ${VENV_PATH}/bin/activate
cd ${WORKSPACE_DIR}
python3 evaluation_scripts/eval_llm_batches.py --model_name /gemma-3-4b-it
"