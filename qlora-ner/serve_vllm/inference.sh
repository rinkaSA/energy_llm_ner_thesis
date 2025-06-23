#!/bin/bash
#SBATCH --job-name=vllm_client
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_eval_output_logs/ev_%j.out
#SBATCH --error=slurm_eval_output_logs/ev_%j.err

WORKSPACE_DIR="/data/horse/ws/irve354e-energy_llm_ner/energy_ner_llm/qlora-ner/serve_vllm"
VENV_PATH="/data/horse/ws/irve354e-energy_llm_ner/super_weights/universal-ner-py311"

SERVER_URL="http://i8001.alpha.hpc.tu-dresden.de:8000/v1/completions"
echo "→ SERVER_URL=${SERVER_URL}"

export SERVER_URL

srun bash -c "
source ${VENV_PATH}/bin/activate
cd ${WORKSPACE_DIR}
python3 batched_eval.py
"