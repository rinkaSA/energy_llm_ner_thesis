
## 1. Directory Structure

```plaintext
serve_vllm/
├── containers/              -> built singularity containers (git ignored) from the images of subdir /Dockerfiles
├────── /Dockerfiles         -> dockerimages of containers for serve and inference
│   └── vllm_serve_otel.sif  -> singularity build vllm_serve_otel.sif docker://irv12/vllm_serve_otel:latest
│   └── inference.sif
├── gpu_metrics_server/      -> during server slurm job gpu metrics are recorded here with nvi smi
├── models/                  -> llms are stored here (now only g.ignored llama2, later will be used for comressed ones)
│   └── base/
│       └── Llama-2-7b-chat-hf/
├── monitoring/             -> directory for monitoring slurm job with dependencies for correct volume binding and logs
│   ├── prometheus.yml
│   ├── vllm_targets.json   -> for dynamical recognition of server nodes name (for monitoring job)
│   ├── gpu_targets.json 
│   ├── node_targets.json
│   ├── prometheus.log
│   ├── jaeger.log
│   ├── dcgm_exporter.log   
│   └── grafana/
│       ├── data/
│       ├── logs/
│       ├── provisioning/
│       │   ├── datasources/
│       │   └── dashboards/
│       └── dashboards/ 
├── slurm_serve_output_logs/
└── slurm_monitor_output_logs/

┌───────────────────┐       gRPC OTLP      ┌───────────────────┐
│  GPU Node         │ ───────────────────▶ │  Monitor Node     │
│  (vLLM Serve)     │                      │  (Prometheus,     │
│  • vllm serve     │    Metrics scrape    │   Jaeger, Grafana)│
│  • /metrics @8000 │ ◀─────────────────── │                   │
│  • OTLP@4317      │                      │                   │ 
│  • DCGM Exp.@9400 │                      │                   │
└───────────────────┘                      └───────────────────┘


```

## 2. Node Roles & SLURM Jobs

### 2.1 GPU Node (`serve_vllm.sh`)
- **Runs**  
  - The vLLM inference server (`opentelemetry-instrument vllm serve` on port `8000`)  
  - **DCGM-Exporter** on port `9400` (GPU power, util, temp, energy, memory, etc.)  

- **Writes**  
  - `monitoring/vllm_targets.json` → Prometheus file_sd (file-based service discovery)
  - `monitoring/gpu_targets.json` → DCGM file_sd  
- **Logs to**  
  - `gpu_metrics_server/` (nvidia-smi CSV)  
  - `monitoring/*.log`

### 2.2 Monitor Node (`monitor_cpu.sh`)
- **Runs** (all in host networking)  
  - **Prometheus** (container) scraping:  
    - vLLM API (`…:8000/metrics`)  
    - DCGM (`…:9400/metrics`)   
    - Jaeger metrics (`localhost:14269/metrics`)  
  - **Jaeger all-in-one** (collector + query UI) on ports:  
    - Zipkin HTTP → `9411`
    - gRPC → `14250`  
    - Admin/metrics → `14269`  
    -  ! UI/API → `16686`  
  - **Grafana** serving dashboards on `3000`
- **Mounts**  
  - `monitoring/` → `/monitoring` inside Prometheus & Jaeger  
  - `monitoring/grafana/...` → `/var/lib/grafana`, `/var/log/grafana`, `/etc/grafana/provisioning`, etc.
- **Logs to**  
  - `monitoring/prometheus.log`  
  - `monitoring/jaeger.log`  
  - `monitoring/grafana/logs/grafana.log`
  - `monitoring/dcgm_exporter.log`

## 3. Ports & Endpoints

| Service           | Host Port | Inside Container | Path           | Purpose                                         |
|-------------------|-----------|------------------|----------------|-------------------------------------------------|
| vLLM API/metrics  | 8000      | 8000             | `/metrics`     | Inference API + Prometheus metrics endpoint     |
| OTLP Traces       | –         | –                | gRPC:4317      | Jaeger OTLP ingestion from vLLM                 |
| DCGM Exporter     | 9400      | 9400             | `/metrics`     | GPU power/temp/utilization/energy/etc.          |
| to do?! node_expor|           |                  |                |                                                 |
|    ter  9100      | 9100      | `/metrics`       | CPU, memory, load, disks, custom |textfiles                      |
| Prometheus UI     | 9090      | 9090             | `/` & `/graph` | Prometheus dashboard & query UI                 |
| Jaeger metrics    | 14269     | 14269            | `/metrics`     | Jaeger’s Prometheus‐style metrics (admin port)  |
| Jaeger UI & API   | 16686     | 16686            | `/`            | Trace search & visualization UI                 |
| Grafana UI        | 3000      | 3000             | `/`            | Grafana dashboards (vLLM, GPU, node, Jaeger)    |

> **Note:** All container services run with `--network host`, so they bind directly to the node’s TCP stack.

## 4. Inter-Node Communication

1. **GPU → Monitor**  
   - GPU node writes `GPU_NODE:8000/metrics`, `:9400/metrics` (`:9100/metrics` if node exporter implemented for cpu)
   - Monitor’s Prometheus scrapes those endpoints over the HPC network.

2. **Monitor → GPU (Jaeger)**  
   - vLLM serve sends OTLP spans via `grpc://MON_NODE:4317`.  
   - Jaeger collector on monitor listens on 4317, ingests spans.

3. **Local ↔ Monitor (User)**  
   - From your laptop: SSH tunnel (`-L3000:localhost:3000 -L9090:localhost:9090 -L16686:localhost:16686`)  
   - Browse Grafana, Prometheus, and Jaeger UIs as if they were local.

## 5. Usage Notes

- **Order** :
  1. `sbatch monitor_cpu.sh`
  2. `sbatch serve_vllm_model.sh`
  3. `sbatch inference_on_vllm.sh` - uses the address of gpu node and endpoint of vlm for generation to send the requests there
  4. `ssh -J {user}@login2.{partittion}.hpc.tu-dresden.de \
      -L9090:localhost:9090 \
      -L3000:localhost:3000 \
      -L16686:localhost:16686 \
      {user}@{monitor node}.{partition}.hpc.tu-dresden.de`
  5. all those ports became available via browser. check out @3000 for grafana - the most important one.


- **Dynamic GPU node discovery**  
  - GPU script writes three target files → Prometheus auto‐reload via `file_sd_configs`.  
  - No need to reconfigure Prometheus when GPU job lands on a new host. !But for now if monitor job is reloaded need to restart vllm job as well!

- **Data persistence**  
  - Grafana’s SQLite DB lives in `monitoring/grafana/data/grafana.db`.  
  - Prometheus TSDB lives in its default path under `/monitoring`.

- **Credentials & security**  
  - Grafana admin password set via `SINGULARITYENV_GF_SECURITY_ADMIN_PASSWORD` on first run.  
  - No authentication on Prometheus or Jaeger by default—access protected by SSH tunnel / firewall.

- **Extending exporters**  
  - To trim DCGM metrics, bind‐mount a custom CSV (`custom-counters.csv`) into `/etc/dcgm-exporter/` and pass `-f` (to do?).  

## 6. Metrics

https://docs.nvidia.com/datacenter/dcgm/latest/user-guide/feature-overview.html#cpu-and-core-fields
https://docs.byteplus.com/en/docs/vmp/Common-metrics-of-DCGM

DCGM talks directly to the NVIDIA driver’s host engine (nv-hostengine) to query these hardware sensors and counters.

- Power Usage                              | DCGM_FI_DEV_POWER_USAGE (Instantaneous board-level power draw, GPU + other power curcuity, in Watts)
- Power Limit                              | DCGM_FI_DEV_POWER_LIMIT (The enforced upper bound (in W) on the device’s power draw, typically set 
                                              to the GPU’s TDP (Termal Design Power) or a user-configured power cap (with nvidia smi flag))
- Energy Consumption                       | DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION (Cumulative energy consumed by the GPU in millijoules (mJ) since the last driver load)

- GPU Utilization                          | DCGM_FI_DEV_GPU_UTIL (Percent of time over the sampling period that one or more streaming multiprocessors were actively executing instructions)
- GPU Memory Total / Used                  | Total VRAM ≈ FB_USED + FB_FREE, DCGM_FI_DEV_FB_USED, DCGM_FI_DEV_FB_FREE
- GPU Temperature                          | DCGM_FI_DEV_MEMORY_TEMP (Current temperature of the on-board GDDR/HBM memory modules in °C.). This is separate from the GPU core temperature (DCGM_FI_DEV_GPU_TEMP)

- Request Count                            | Provided by vLLM (vllm:request_success_total)
- Execution Count                          | Same as above + vllm:num_requests_running
- Inference Count                          | vllm:generation_tokens_total
- Latency (Request / Compute / Queue Time) | vLLM histograms (vllm:e2e_request_latency_seconds_bucket,
                                             vllm:request_prefill_time_seconds_bucket, etc.)
- CPU Utilization & Temperature            | via node_exporter (to do)
