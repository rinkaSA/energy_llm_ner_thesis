# Server, monitoring and inference pass


## Current usage of two and one node serving+monitoring

Earlier we relied on the consecutive job runnning of two separate job on two different nodes. 
That is described in the [text](README_two_nodes.md) file. (basically 2 `sbatch`` of two .sh files where needed)

Now we want to build it in one node, but share the CPU resources reserved for monitor and for serve (to acccurately measure the latter).

For that we work on the 'one_node_serve.sh' file.
At the moment the desired behavior of running in parallel two tasks defined by 'srun' while deviding resources.

Here are some steps needed to be done before running it for another user.

# 1. Pull docker container desined for vllm serve.
Put it into ```./serve_vllm/containers``` directory.

```singularity build vllm_serve_otel.sif docker://irv12/vllm_serve_otel:latest```

{Docker for inference:
```singularity build inference.sif docker://irv12/inference:latest```} -> deprecated, build a VENV from requirements and use it for inference!

# 2. Download LLama 2

Prerequsites are having account on hugging face where the usage terms of Meta where accecpted (https://medium.com/@tushitdavergtu/how-to-install-llama-2-locally-d3e3c6c8eb4c)

- Install /log into HF.

```pip install huggingface_hub ```

```huggingface-cli login```

- And then clone it

```git clone https://huggingface.co/meta-llama/Llama-2-7b-chat-hf```

Currently the path to the model is:
```./energy_ner_llm/qlora-ner/serve_vllm/models/base/Llama-2-7b-chat-hf```

```/models``` directory is not on the remote since all the large stuff was gitignored. Create it manually or adjust pathes accordingly.
# 3. Check all the pathes in one_node_serve.sh - change the base one.

Adjust your working directory path!!!

# 4. Run with sbatch one_node_serve.sh

```sbatch one_node_serve.sh```


Check for the node name where it is running

```squeue -u <user_name>``` 

also check for the logs in the dir ```/slurm_onenode``` and all ```.log``` files in ```/monitoring```

# 5. If you want to submit a request to this server....

- Change in .env (create if not done already) ```SERVER_URL```  to the current path (node) where the model is hosted. Also, to be sure I statically added the variable with this path to ```inference.sh```. Adjust the pathes to your working directory as well as servers url (e.g. SERVER_URL="http://i8001.alpha.hpc.tu-dresden.de:8000/v1/completions")

- Run ```sbatch inference.sh```.


# Details on inference
This experiment evaluates the trade-offs between throughput, latency and energy efficiency when running batched NER inference on a vLLM server. We vary the client-side batch size and capture both model performance (precision/recall/F1) and detailed telemetry (latency, token counts, energy consumption) per batch, logging everything to MLflow for easy comparison.


- **Fine-grained energy analysis**: by measuring energy and tokens on the client side, we omit coarse scrape intervals (in prometheus - 15 sec) and directly attribute joules to each inference batch.

- **Phase-aware profiling**: combining DCGM power traces with vLLM’s prefill vs. decode histograms lets us decompose where the GPU spends most energy—prompt processing or token generation.

- **Batch sizing insights**: sweeping through batch sizes reveals the sweet spot that maximizes throughput without incurring undue latency or energy inefficiency.

### The aim is:
- See trade-offs: Small batches finish quickly but waste more energy per token; big batches use energy more efficiently but take longer to get the first result.

- Find the best point: By comparing runs, we could discover the batch size that gives good speed without using too much extra power.

- Understand where energy goes: Splitting “prefill” (reading the prompt) and “decode” (generating tokens) shows whether most energy is spent loading the prompt or making new tokens.


## 1. Structure & Workflow
### 1. Data Preparation

Load the CoNLL-2003 test split (first N examples). - for my experiments I take 320 (because I will be looking deeper on [32,64,128] batch sizes, and wanted to see how will 32*10 examples - 10 batches - work out)

### 2. Extract tokens and gold NER tags.

### 3. Batch Sweep Loop
```
for B in [1, 4, 8, 16, 32, 64, 128]:
    with mlflow.start_run(run_name=f"batch_{B}"):
        evaluate_ner_pipeline_conll03(batch_size=B)
``` 
For each batch size B, we start a separate MLflow run so ``` metrics/```  artifacts are organized per configuration.

### 4. Asynchronous Inference

- A single HTTP-based coroutine (single_request) sends one prompt → one LLM call.

- process_batch fans out B of these coroutines in parallel via asyncio.gather, simulating client-side batching.

### 5. Telemetry & Instrumentation

- Time: stamped immediately before and after the await process_batch(...) with time.perf_counter().

- vLLM Token Counters: scraped from the server’s Prometheus /metrics endpoint before/after each batch to get exact prompt/generation token totals.

- Energy: 

    Cumulative joules from DCGM’s TOTAL_ENERGY_CONSUMPTION counter (in miliJoules!) or

    (we could optinally do : Estimated joules = average POWER_USAGE (W) × batch latency (s)) 

### 6. NER Metric Computation

- Parse the JSON‐style responses into entity tags.

- Compute precision, recall, F1 and per‐class breakdown with seqeval.

- Logging to MLflow

-Parameters: batch_size, prompt template.

-Metrics (PER BATCH):

    - NER: precision / recall / F1 / per‐class scores  

    - Telemetry per batch for ENERGY LATENCCY ANALYSIS: latency_s, energy_j, prompt_tokens, generation_tokens, total_tokens, joules_per_token, plus vLLM‐derived means (E2E, TTFT, per-token) - .

- Artifacts: raw JSON of generated outputs and the full telemetry list.

### 7. Notes and recomendations
Visit MLflow UI to compare per-batch runs side by side.

Use Grafana to verify GPU‐side metrics (power, cache utilization, scheduler stats).

