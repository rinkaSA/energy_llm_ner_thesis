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

HOST_BASE="/data/horse/ws/irve354e-energy_llm_ner/energy_ner_llm/qlora-ner/serve_vllm"
HOST_MON="${HOST_BASE}/monitoring"
HOST_REPO="${HOST_BASE}"
INFERENCE_SIF="${HOST_BASE}/containers/inference.sif"

GPU_NODE=$(< "${HOST_MON}/gpu_node.txt")
SERVER_URL="http://${GPU_NODE}:8000/v1/completions"
echo "→ SERVER_URL=${SERVER_URL}"

export SINGULARITYENV_SERVER_URL="${SERVER_URL}"


srun singularity exec \
    -B "${HOST_REPO}":/workspace:rw\
    -B "${HOST_MON}":/monitoring:ro \
    --pwd /workspace \
    "${INFERENCE_SIF}" \
    python3 eval_llm_batches.py

