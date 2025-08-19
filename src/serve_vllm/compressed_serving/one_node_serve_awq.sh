#!/bin/bash
#SBATCH --job-name=vllm_stack
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6  # was 8 for capella
#SBATCH --gres=gpu:1        
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_onenode/slurm_%x_%j.out
#SBATCH --error=slurm_onenode/slurm_%x_%j.err


BASE="/data/horse/ws/irve354e-energy_llm_ner/energy_ner_llm/src"
HOST_BASE="${BASE}/serve_vllm"

HOST_MON="${HOST_BASE}/monitoring"
HOST_MODEL="${BASE}/models/base/Mistral-7B-Instruct-v0.2-AWQ"
SIF_IMAGE="${HOST_BASE}/containers/vllm_serve_otel.sif"

mkdir -p "${HOST_MON}"/{prometheus,jaeger,grafana/{data,logs,provisioning,dashboards}}
srun --overlap -n1 --cpus-per-task=3 --cpu-bind=cores bash <<EOF &
  set -x

  # GPU metrics exporter
  singularity exec --nv --network host \
    docker://nvidia/dcgm-exporter:3.1.7-3.1.4-ubuntu20.04 \
    /usr/bin/dcgm-exporter --address=:9400 \
    --c 500 \
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

  # Record GPU metrics
  nvidia-smi \
    --query-gpu=timestamp,utilization.gpu,utilization.memory,temperature.gpu \
    --format=csv -l 5 \
    > "${HOST_BASE}/gpu_metrics_server/gpu_metrics_${SLURM_JOB_NAME}.csv" &
  gpu_log_pid=\$!

  # Start vLLM server
  singularity exec --nv --network host \
    -B "${HOST_MODEL}":/model:ro \
    -B "${HOST_MON}":/monitoring \
    -B "${HOST_BASE}/templates":/templates:ro \
    "${SIF_IMAGE}" \
      opentelemetry-instrument vllm serve /model \
        --host 0.0.0.0 --port 8000 \
        --quantization awq \
        --chat-template /templates/mistral.jinja \
        --tensor-parallel-size 1 \
        --max-num-batched-tokens 32768 \
        --max-num-seqs 256 \
        --block-size 16 \
        --otlp-traces-endpoint="grpc://localhost:4317" \
    > "${HOST_MON}/vllm.log" 2>&1

  kill \${gpu_log_pid}
EOF

wait