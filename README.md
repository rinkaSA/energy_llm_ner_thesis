# Energy Consumption Measurement for Multilingual NER with Large Language Models

This repository contains code and experiments for measuring the energy efficiency of various Large Language Models (LLMs) when performing Named Entity Recognition (NER) tasks across multiple languages. The experiments focus on comparing Mistral, Gollie, and Gemma models while analyzing their energy consumption per token and total energy usage in joules.

## Repository Structure

```
energy_llm_ner/
├── energy_llm_ner_thesis/        # Main project code
│   ├── src/                      # Source code
│   │   ├── serve_vllm/           # VLLM serving and metrics collection
│   │   │   ├── collect_gpu_nvml.py     # Script to collect GPU metrics via NVML
│   │   │   ├── inference_base.sh       # Base script to initialize metrics for most models
│   │   │   ├── inference_base_gollie.sh # Base script to initialize metrics for Gollie model
│   │   │   ├── inference_gemma.sh      # Script to run inference with Gemma across languages
│   │   │   ├── metrics_exporter.py     # Exports metrics from VLLM for collection
│   │   │   └── serve_model.py          # Script to serve models using VLLM
│   │   │
│   │   └── xtreme/               # XTREME dataset handling and inference scripts
│   │       ├── dataset_utils.py        # Utilities for processing the XTREME dataset
│   │       ├── inference.py            # Main inference logic for NER tasks
│   │       ├── language_specific/      # Language-specific configurations
│   │       ├── ner_prompts.py          # NER prompting templates
│   │       └── result_processing.py    # Process and evaluate NER results
│   │
│   ├── data/                     # Data storage
│   │   ├── xtreme/                     # XTREME dataset
│   │   └── metrics/                    # Collected energy and performance metrics
│   │
│   └── results/                  # Results and analysis
│       ├── energy/                     # Energy consumption measurements
│       └── ner/                        # NER performance results
```

## Experimental Setup

### Task Description
The main task is **multilingual Named Entity Recognition (NER)** using the XTREME dataset. The goal is to evaluate how different LLMs perform on this task across multiple languages while measuring their energy consumption.

### Models Evaluated
- **Mistral**: A state-of-the-art language model
- **Gollie**: A multilingual language model
- **Gemma**: Google's lightweight language model

### Languages
The experiments are conducted across multiple languages from the XTREME dataset to evaluate the models' multilingual capabilities.

## Energy Measurement Methodology

Energy consumption is measured using NVIDIA's Data Center GPU Manager (DCGM) metrics, collected through the `collect_gpu_nvml.py` script. Key metrics include:

- GPU utilization
- Memory usage
- Temperature
- Power consumption
- Total energy consumption in joules

These metrics are collected at regular intervals (configurable via `METRICS_INTERVAL_MS`) during model inference.

## Running Experiments

### Prerequisites
1. Set up environment variables:
   - `ENERGY_URL`: URL to access GPU metrics
   - `METRICS_CSV`: Path to save metrics data
   - `METRICS_INTERVAL_MS`: Sampling interval in milliseconds (default: 1000ms)

### Step 1: Initialize VLLM Metrics Collection
Before running inference, you must initialize the Grafana dashboard and VLLM metrics collection. Use one of the following scripts depending on the model:

```bash
# For most models
./src/serve_vllm/inference_base.sh

# For Gollie model
./src/serve_vllm/inference_base_gollie.sh
```

These scripts set up the necessary environment for metrics collection before the actual inference takes place.

### Step 2: Run Inference
After initializing metrics collection, you can run the inference experiments:

```bash
# For Gemma across all languages
./src/serve_vllm/inference_gemma.sh
```

The scripts in the `xtreme` folder handle sending requests to the models using the XTREME dataset for NER tasks.

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
