# Energy Consumption Measurement for Multilingual NER with Large Language Models

This repository contains code and experiments for measuring the energy efficiency of various Large Language Models (LLMs) when performing Named Entity Recognition (NER) tasks across multiple languages. The experiments focus on comparing Mistral, Gollie, and Gemma models while analyzing their energy consumption per token and total energy usage in joules.

## Repository Structure

```

energy_llm_ner_thesis/       
   ├── src/            
   │   ├── serve_vllm/           # VLLM serving and metrics collection
   │   │   ├── collect_gpu_nvml.py     # Script to collect GPU metrics via NVML
   │   │   ├── inference_base.sh       # Base script to initialize metrics for most models
   │   │   ├── inference_base_gollie.sh # Base script to initialize metrics for Gollie model
   │   │   ├── inference_gemma.sh      # Script to run inference with Gemma across languages
   │   │   ├── metrics_exporter.py     # Exports metrics from VLLM for collection
   │   │   └── serve_model.py          # Script to serve models using VLLM
   │   │
   │   └── xtreme/               # XTREME dataset handling and inference scripts
   │       ├── dataset_utils.py        # Utilities for processing the XTREME dataset
   │       ├── inference.py            # Main inference logic for NER tasks
   │       ├── language_specific/      # Language-specific configurations
   │       ├── ner_prompts.py          # NER prompting templates
   │       └── result_processing.py    # Process and evaluate NER results
   │
   ├── data/                     # Data storage
   │   ├── xtreme/                     # XTREME dataset
   │   └── metrics/                    # Collected energy and performance metrics
   │
   └── results/                  # Results and analysis
       ├── energy/                     # Energy consumption measurements
       └── ner/                        # NER performance results
```

## Experimental Setup

## Main workflow idea
There are 3 scripts that form a cohesive experimental workflow for evaluating multilingual NER under energy-aware conditions:
- /src/serve_vllm/SERVE_model_exec/<model_name>/one_node_serve_<model_name>.sh: Sets up the backbone — a vLLM server hosting the Mistral/Geema 3/ GoLLIE model, wrapped with a complete monitoring stack (Prometheus, Grafana, Jaeger, DCGM) to track GPU utilization and energy metrics in real time. I use H100-80GB one node, so the models are up to 12B to fit it nicely.

- /src/serve_vllm/SERVE_model_exec/<model_name>/inference_<model>_all_langs.sh: Acts as the orchestrator — running evaluation loops across different languages and prompt settings, while ensuring GPU metrics are continuously collected and experiments are paced with cooldowns.

- /src/serve_vllm/evaluation_scripts/xtreme/lang_eval_mlflow_mi_gol.py: (run inside previous bash script) Delivers the analysis — executing the actual NER benchmarking, parsing model outputs, computing precision/recall/F1, attributing energy costs, and logging everything to MLflow for reproducibility.

Together, they connect serving, evaluation, and monitoring into a unified pipeline for energy-efficient large language model experimentation.

### Task Description
The main task is **multilingual Named Entity Recognition (NER)** using the XTREME/ Masakha dataset. The goal is to evaluate how different LLMs perform on this task across multiple languages while measuring their energy consumption. At the moment I concentrate on XTREME dataset.

### Models Evaluated
- **Mistral-7B-Instruct-v0.2**: [HF](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)
- **Gollie-7B**: [HF](https://huggingface.co/HiTZ/GoLLIE-7B) (based on Llama 2 Code)
- **Gemma-3-4b/12b-it**: [HF](https://huggingface.co/google/gemma-3-4b-it)

### Languages
The experiments are conducted across multiple languages from the XTREME dataset to evaluate the models' multilingual capabilities. We are going to evaluate English and German for 8 different prompts, but consider only one type of a prompt for all languages evaluation. For brevity, the showcase languages will be Italian - it, German - de, Englisch - en, Bulgarian - bg, Chinese - zh.

### Energy Measurement Methodology

Energy consumption is measured using NVIDIA's Data Center GPU Manager (DCGM) metrics, scraped through the `collect_gpu_nvml.py` script of DCGM exposed port 9400 with GPU's metrics (100ms).

Key metrics include:

- GPU utilization, %
- Memory usage, MB
- Temperature,  C
- Power consumption, Watts
- Total energy consumption since boot, Joules

## What the Serving Job (vllm_stack) Does

- SLURM resources: 1 node, 1 GPU, 6 CPUs, 64 GB RAM, 24 h max.

   - Note: When the job is sent with SLURM to the HPC (of TU Dresden), the node name and resources are allocated dynamically. We store the node name of servin gjob to use it for fursther request posting with inference jobs.

- Environment: Resolves node name and writes NODE_NAME=… to ${ENV_FILE} (make sure ${ENV_FILE} exists or is exported beforehand).

- Monitoring stack (all via Singularity with --network host)

 - NVIDIA DCGM Exporter (GPU metrics)
      Image: docker://nvidia/dcgm-exporter:3.1.7-3.1.4-ubuntu20.04
      Args: --address=:9400 --collect-interval=100
      Output log: ${HOST_MON}/dcgm.log

- Prometheus
      Image: docker://prom/prometheus:v3.4.0
      Expects config at /monitoring/prometheus.yml (bind from ${HOST_MON})
      Log: ${HOST_MON}/prometheus.log
   
 - Jaeger All-in-One
   Image: docker://jaegertracing/all-in-one:1.57
   Zipkin ingress: --collector.zipkin.host-port=9411
   Log: ${HOST_MON}/jaeger.log

-  Grafana
   Image: docker://grafana/grafana:latest
   Env (inside container):
   GF_SECURITY_ADMIN_PASSWORD=admin, GF_DASHBOARDS_JSON_ENABLED=true
   Binds:
   
   /var/lib/grafana ↔ ${HOST_MON}/grafana/data
   
   /var/log/grafana ↔ ${HOST_MON}/grafana/logs
   
   /etc/grafana/provisioning (RO) ↔ ${HOST_MON}/grafana/provisioning
   
   /var/lib/grafana/dashboards (RO) ↔ ${HOST_MON}/grafana/dashboards
   Log: ${HOST_MON}/grafana.log

   - vllm container: singularity build vllm_serve_otel.sif docker://irv12/vllm_serve_otel:latest 
      vllm serve model_name \
     --host 0.0.0.0 --port 8000 \
     --tensor-parallel-size 1 \
     --dtype bfloat16 \
     --kv-cache-dtype fp8 --calculate-kv-scales \
     --gpu-memory-utilization 0.95 \
     --enable-prefix-caching \
     --enable-chunked-prefill \
     --block-size 16 \
     --max-model-len 4096 \
     --max-num-batched-tokens 49152 \
     --max-num-seqs 128 \
     --otlp-traces-endpoint grpc://localhost:4317
   Model: ${HOST_MODEL}:/mistral:ro # be sure to have the mdoel locally to bind it to the serving container
   
   Monitoring: ${HOST_MON}:/monitoring 
   
   Templates: ${HOST_BASE}/templates:/templates:ro
   
   Log: ${HOST_MON}/vllm.log
   - 

## Running Experiments

### Step 1. Prerequisites
 Set up environment variables in /serve_vllm/.env file (template_provided)
   - `CHAT_URL` \ `COMPLETION_URL` - where to post requests based on the model type (instructiong tuned use chat endpoint, while generational on euse compeltions)
   - `ENERGY_URL`: URL to access GPU metrics (constructed based on the node name where th emodel is hosted and monitored, the same as previous)
   - `METRICS_CSV`: Path to save metrics data, where to store it, this will be logged to mlflow during each experiment
   - `METRICS_INTERVAL_MS`: Sampling interval in milliseconds (default: 100ms as in DCGM exporter)
   - `MLFLOW_URL` (user and password as well) - to log experiment.

### Step 2. Host the model

```cd src/serve_vllm```
``` sbatch SERVE_model_exec/gemma/one_node_serve_gemma.sh```
### Step 3. CHeck if the .env file have a node name updated based on the assigned ndoe for the previous task.

Per-run loop

For each (language, prompt_style):

Chooses max_new_tokens by style (here: style 6 → 100).

Records a UTC start timestamp: YYYY-MM-DDTHH:MM:SS.ffffff.

Executes the evaluator with srun inside the venv:

python evaluation_scripts/xtreme/lang_eval_mlflow_mi_gol.py \
  --language <lang> \
  --model /mistral \
  --system-prompt-choice <style> \
  --batch-size 128 \
  --max-new-tokens <per style> \
  --limit 10000 \
  --run-start-timestamp "<UTC ISO ts>" \
  --use-generic-template
## Energy Efficiency Analysis

The collected metrics are used to analyze two key aspects of energy efficiency:

1. **Energy per token**: How much energy (in joules) is required to process each token
2. **Total energy consumption**: The overall energy used during the entire inference process

## Results

### Energy Consumption

| Model  | Language | Energy per Token (J) | Total Energy (J) | Inference Time (s) |
|--------|----------|---------------------|-----------------|-------------------|
|        |          |                     |                 |                   |
|        |          |                     |                 |                   |
|        |          |                     |                 |                   |

### NER Performance

| Model  | Language | Precision | Recall | F1 Score |
|--------|----------|-----------|--------|----------|
|        |          |           |        |          |
|        |          |           |        |          |
|        |          |           |        |          |

## Analysis and Conclusions

(This section will be updated after experiments are completed with analysis of the energy consumption patterns and their correlation with model performance across different languages.)
