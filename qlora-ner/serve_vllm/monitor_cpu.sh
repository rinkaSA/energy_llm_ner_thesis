#!/bin/bash
#SBATCH --job-name=monitor-stack
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_monitor_output_logs/mon_%j.out
#SBATCH --error=slurm_monitor_output_logs/mon_%j.err

HOST_MON_PATH="/data/horse/ws/irve354e-energy_llm_ner/energy_ner_llm/qlora-ner/serve_vllm/monitoring"

mkdir -p "${HOST_MON_PATH}"

hostname --fqdn > "${HOST_MON_PATH}/monitor_node.txt"
echo "Monitoring on $(< "${HOST_MON_PATH}/monitor_node.txt")"

echo "[]" > "${HOST_MON_PATH}/vllm_targets.json"

singularity exec --network host \
  -B "${HOST_MON_PATH}":/monitoring \
  docker://prom/prometheus:latest \
  prometheus --config.file=/monitoring/prometheus.yml \
  > "${HOST_MON_PATH}/prometheus.log" 2>&1 &

singularity run --network host \
  -B "${HOST_MON_PATH}":/monitoring \
  docker://jaegertracing/all-in-one:1.57 \
    --collector.zipkin.host-port=9411 \
  > "${HOST_MON_PATH}/jaeger.log" 2>&1 &

# GPU metrics exporter on port 9400 !! ( on serve node, not here)


export SINGULARITYENV_GF_SECURITY_ADMIN_PASSWORD="admin"
export SINGULARITYENV_GF_DASHBOARDS_JSON_ENABLED="true"

singularity exec --network host \
  -B "${HOST_MON_PATH}/grafana/data":/var/lib/grafana:rw \
  -B "${HOST_MON_PATH}/grafana/logs":/var/log/grafana:rw \
  -B "${HOST_MON_PATH}/grafana/provisioning":/etc/grafana/provisioning:ro \
  -B "${HOST_MON_PATH}/grafana/dashboards":/var/lib/grafana/dashboards:ro \
  docker://grafana/grafana:latest \
  /run.sh \
  > "${HOST_MON_PATH}/grafana/logs/grafana.log" 2>&1 &

wait
