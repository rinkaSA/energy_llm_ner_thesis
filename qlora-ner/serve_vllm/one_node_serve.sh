#!/bin/bash
#SBATCH --job-name=vllm_stack
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6     
#SBATCH --gres=gpu:1        
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_%x_%j.out
#SBATCH --error=slurm_%x_%j.err

HOST_BASE="/data/horse/ws/irve354e-energy_llm_ner/energy_ner_llm/qlora-ner/serve_vllm"
HOST_MON="${HOST_BASE}/monitoring"
HOST_MODEL="${HOST_BASE}/models/base/Llama-2-7b-chat-hf"
SIF_IMAGE="${HOST_BASE}/containers/vllm_serve_otel.sif"

mkdir -p "${HOST_MON}"/{prometheus,jaeger,grafana/{data,logs,provisioning,dashboards}}

srun --overlap -n1 --cpus-per-task=2 --cpu-bind=cores bash <<EOF &
  set -x

  singularity exec  --nv --network host \
    docker://nvidia/dcgm-exporter:3.1.7-3.1.4-ubuntu20.04 \
    /usr/bin/dcgm-exporter --address=:9400 \
    > "${HOST_MON}/dcgm.log" 2>&1 &

  singularity exec --network host \
    -B "${HOST_MON}":/monitoring:rw \
    docker://prom/prometheus:v3.4.0 \
    prometheus --config.file=/monitoring/prometheus.yml \
    > "${HOST_MON}/prometheus.log" 2>&1 &

  singularity run  --network host \
    -B "${HOST_MON}":/monitoring:rw \
    docker://jaegertracing/all-in-one:1.57 \
      --collector.zipkin.host-port=9411 \
    > "${HOST_MON}/jaeger.log" 2>&1 &

  singularity exec --network host \
    -B "${HOST_MON}/grafana/data":/var/lib/grafana:rw \
    -B "${HOST_MON}/grafana/provisioning":/etc/grafana/provisioning:ro \
    -B "${HOST_MON}/grafana/dashboards":/var/lib/grafana/dashboards:ro \
    docker://grafana/grafana:latest /run.sh \
    > "${HOST_MON}/grafana.log" 2>&1 &

  sleep 5
EOF

srun --overlap -n1 --cpus-per-task=4 --cpu-bind=cores --gres=gpu:1 bash <<EOF
  set -x

  singularity exec --nv --network host \
    -B "${HOST_MODEL}":/model:ro \
    -B "${HOST_MON}":/monitoring \
    "${SIF_IMAGE}" \
      opentelemetry-instrument vllm serve /model \
        --host 0.0.0.0 --port 8000 \
        --otlp-traces-endpoint="grpc://localhost:4317" \
    > "${HOST_MON}/vllm.log" 2>&1
EOF

wait
