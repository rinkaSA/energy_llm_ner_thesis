#!/bin/bash
#SBATCH --job-name=vllm_stack
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6  # was 8 for capella
#SBATCH --gres=gpu:1        
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_out/slurm_onenode/slurm_%x_%j.out
#SBATCH --error=slurm_out/slurm_onenode/slurm_%x_%j.err

BASE="/data/horse/ws/irve354e-energy_llm_ner/energy_ner_llm/src"
HOST_BASE="${BASE}/serve_vllm"

MODEL_NAME="gemma-3-4b-it"
HOST_MON="${HOST_BASE}/monitoring"
HOST_MODEL="${BASE}/models/base/gemma-3-4b-it"
SIF_IMAGE="${HOST_BASE}/containers/vllm_serve_otel.sif"
ENV_FILE="${HOST_BASE}/.env"



mkdir -p "${HOST_MON}"/{prometheus,jaeger,grafana/{data,logs,provisioning,dashboards}}
srun --overlap -n1 --cpus-per-task=3 --cpu-bind=cores bash <<EOF &
  set -x

  # GPU metrics exporter
  singularity exec --nv --network host \
    docker://nvidia/dcgm-exporter:3.1.7-3.1.4-ubuntu20.04 \
    /usr/bin/dcgm-exporter --address=:9400 \
    --collect-interval=100 \
    > "${HOST_MON}/dcgm.log" 2>&1 &

  # Prometheus
  singularity exec --network host \
    -B "${HOST_MON}":/monitoring:rw \
    docker://prom/prometheus:v3.4.0 \
    prometheus --config.file=/monitoring/prometheus.yml \
    > "${HOST_MON}/prometheus.log" 2>&1 &

  # Jaeger
  singularity run --network host \
    -B "${HOST_MON}":/monitoring:rw \
    docker://jaegertracing/all-in-one:1.57 \
      --collector.zipkin.host-port=9411 \
    > "${HOST_MON}/jaeger.log" 2>&1 &

  # Grafana
  export SINGULARITYENV_GF_SECURITY_ADMIN_PASSWORD="admin"
  export SINGULARITYENV_GF_DASHBOARDS_JSON_ENABLED="true"
  
  singularity exec --network host \
    -B "${HOST_MON}/grafana/data":/var/lib/grafana:rw \
    -B "${HOST_MON}/grafana/logs":/var/log/grafana:rw \
    -B "${HOST_MON}/grafana/provisioning":/etc/grafana/provisioning:ro \
    -B "${HOST_MON}/grafana/dashboards":/var/lib/grafana/dashboards:ro \
    docker://grafana/grafana:latest \
    /run.sh \
    > "${HOST_MON}/grafana.log" 2>&1

  wait
EOF


sleep 5


srun --overlap -n1 --cpus-per-task=3 --cpu-bind=cores --gres=gpu:1 bash <<EOF #was 5!!
  set -x

  export VLLM_LOG_LEVEL=DEBUG
  # Disable torch dynamo compilation to avoid gcc errors
  export TORCH_COMPILE_DEBUG=1
  export TORCH_DYNAMO_DISABLE=1
  export TORCHDYNAMO_DISABLE=1

  # Record GPU metrics
  nvidia-smi \
    --query-gpu=timestamp,utilization.gpu,utilization.memory,temperature.gpu \
    --format=csv -l 5 \
    > "${HOST_BASE}/gpu_metrics_server/gpu_metrics_${SLURM_JOB_NAME}.csv" &
  gpu_log_pid=\$!

  # Start vLLM server
  singularity exec --nv --network host \
    -B "${HOST_MODEL}":/${MODEL_NAME}:ro \
    -B "${HOST_MON}":/monitoring \
    -B "${HOST_BASE}/templates":/templates:ro \
    "${SIF_IMAGE}" \
    opentelemetry-instrument vllm serve "/${MODEL_NAME}" \
      --host 0.0.0.0 --port 8000 \
      --tensor-parallel-size 1 \
      --dtype bfloat16 \
      --kv-cache-dtype fp8 \
      --calculate-kv-scales \
      --gpu-memory-utilization 0.95 \
      --enable-prefix-caching \
      --enable-chunked-prefill \
      --block-size 16 \
      --max-model-len 4096 \
      --max-num-batched-tokens 49152 \
      --max-num-seqs 128 \
      --otlp-traces-endpoint="grpc://localhost:4317" \
    > "${HOST_MON}/vllm.log" 2>&1
EOF

wait