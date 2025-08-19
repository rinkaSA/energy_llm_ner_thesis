#!/bin/bash
#SBATCH --job-name=vllm_server
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=slurm_serve_output_logs/vllm_server_%j.out
#SBATCH --error=slurm_serve_output_logs/vllm_server_%j.err

HOST_BASE="/data/horse/ws/irve354e-energy_llm_ner/energy_ner_llm/qlora-ner/serve_vllm"
HOST_MON="${HOST_BASE}/monitoring"
HOST_LOGS="${HOST_BASE}/gpu_metrics_server"
HOST_MODEL="${HOST_BASE}/models/base/Llama-2-7b-chat-hf"
SIF_IMAGE="${HOST_BASE}/containers/vllm_serve_otel.sif"


mkdir -p "${HOST_LOGS}"

# Register this GPU node for Prometheus (file_sd) 
GPU_FQDN=$(hostname --fqdn)
cat <<EOF > "${HOST_MON}/vllm_targets.json"
[
  {
    "labels": {"job":"vllm"},
    "targets": ["${GPU_FQDN}:8000"]
  }
]
EOF
echo "→ Prometheus target written: ${GPU_FQDN}:8000"

echo "${GPU_FQDN}" > "${HOST_MON}/gpu_node.txt"

cat <<EOF > "${HOST_MON}/gpu_targets.json"
[
  {
    "labels": {"job":"gpu_hw"},
    "targets": ["${GPU_FQDN}:9400"]
  }
]
EOF


# Read the monitor node’s hostname (written by monitor_cpu.slurm)
MON_NODE=$(< "${HOST_MON}/monitor_node.txt")
echo "→ Monitor node (Jaeger/Prometheus): ${MON_NODE}"

nvidia-smi \
  --query-gpu=timestamp,utilization.gpu,utilization.memory,temperature.gpu \
  --format=csv -l 5 \
  > "${HOST_LOGS}/gpu_power.csv" &
gpu_log_pid=$!

singularity exec --network host --nv \
  docker://nvidia/dcgm-exporter:3.1.7-3.1.4-ubuntu20.04 \
    /usr/bin/dcgm-exporter \
      --address=:9400 \
  > "${HOST_MON}/dcgm_exporter.log" 2>&1 &
  
singularity exec --nv --network host \
  -B "${HOST_MODEL}":/model \
  -B "${HOST_MON}":/monitoring \
  -B "${HOST_LOGS}":/logs \
  "${SIF_IMAGE}" \
    opentelemetry-instrument vllm serve \
      /model \
      --host 0.0.0.0 \
      --port 8000 \
      --otlp-traces-endpoint="grpc://${MON_NODE}:4317"


kill ${gpu_log_pid}
