# LLM Inference & Energy Efficiency Metrics

This document outlines the end-to-end flow of an inference request through a vLLM server, the key performance and energy metrics to monitor, how to compute energy efficiency (Joules per token), and how to size batching parameters based on GPU memory.

---

## 1. Inference Request Cycle

1. **Client → HTTP Request**\
   The client sends a JSON payload with:
   ```json
   {
     "model": "llama2-7b",
     "prompt": "<instruction + sentence>",
     "max_tokens": 150
   }
   ```
2. **Server Queueing**
   - The vLLM engine enqueues the request.
   - **Metric:** `vllm:request_queue_time_seconds_*` (time waiting before batching).

3. **Prefill (Prompt Processing)**
   - Tokenization, embedding lookup, KV‑cache population on GPU by each token in prompt.
   - **Metric:** `vllm:request_prefill_time_seconds_*`.
4. **Batch Scheduling**
   - Ray engine groups many requests into one GPU batch (bounded by `--max-num-batched-tokens`).
5. **Inference (GPU Execution)**
   - Model generates output tokens in parallel for the entire batch.
   - **Metric:** `vllm:request_inference_time_seconds_*` or sum of `vllm:request_decode_time_seconds_*` + internal decode time.
6. **Streaming Output**
   - Tokens emitted in blocks of size `--block-size`.
   - **Metrics:**
     - **Time to First Token (TTFT):** `vllm:time_to_first_token_seconds_*`
     - **Inter‑Token Latency:** `vllm:time_per_output_token_seconds_*`
7. **Client Receives Response**
   - Final assembled text returned over HTTP.

---

## 2. Core vLLM Performance Metrics

| Metric                  | Prometheus Name                          | Why It Matters                          |
| ----------------------- | ---------------------------------------- | --------------------------------------- |
| **TTFT**                | `vllm:time_to_first_token_seconds_*`     | User‐perceived latency to first output. |
| **Inter‑Token Latency** | `vllm:time_per_output_token_seconds_*`   | Streaming smoothness.                   |
| **Prompt Tokens/Sec**   | `rate(vllm:prompt_tokens_total[1m])`     | CPU tokenization throughput.(*1)            |
| **Gen Tokens/Sec**      | `rate(vllm:generation_tokens_total[1m])` | GPU token generation throughput.(*2)        |
| **Request Queue Time**  | `vllm:request_queue_time_seconds_*`      | Back‑pressure before GPU.               |
| **Prefill Time**        | `vllm:request_prefill_time_seconds_*`    | Prompt preprocessing cost.              |
| **Inference Time**      | `vllm:request_inference_time_seconds_*`  | Pure model execution (*3)time.              |
| **Decode Time**         | `vllm:request_decode_time_seconds_*`     | Post-GPU (*4)detokenization.                |
| **Req Success Rate**    | `rate(vllm:request_success_total[1m])`   | Throughput of completed requests.       |

(*1) This is the rate at which vLLM is ingesting and tokenizing input prompts (the “prefill” stage). It reflects mostly CPU work (tokenization, embedding lookups, KV‐cache population over the entire prompt, one token at a time, storing K/V for each token in sequence) and happens before any GPU decoding.

(*2) This is the rate at which the model is actually producing new output tokens (the “decode” stage) on the GPU. It’s the metric one would typically use to size GPU batch and to compute energy‐per‐token.

(*3) Covers the GPU’s pure model execution across the whole batch (forward pass & token sampling)

(*4) Covers the CPU-side post-processing (detokenization, formatting, cleanup) for each individual request.


**Histogram percentiles** (p50/p90/p95/p99) are computed with:

```promql
histogram_quantile(0.95,
  sum by(le) (
    rate(vllm:time_to_first_token_seconds_bucket[5m])
  ))
```

---

## 3. GPU Energy Metrics

| Metric                | DCGM Name                        | Units  | Description                      |
| --------------------- | -------------------------------- | ------ | -------------------------------- |
| **Power Usage**       | `DCGM_FI_DEV_POWER_USAGE`        | Watts  | Instantaneous GPU power draw.    |
| **Cumulative Energy** | `DCGM_FI_DEV_ENERGY_CONSUMPTION` | Joules | Total energy since driver start. |

### Calculating Joules per Token

1. **Energy consumed** over run
   - From cumulative counter:\
     E_GPU = E_end - E_start
   - Or integrate power samples:\
     E_GPU ~ \sum P_{GPU} * delta t
2. **Total generated tokens**:\
   N_gen = increase(vllm:generation_tokens_total[run_interval])\)
3. **Energy efficiency**:\
   J/token = E_GPU/N_gen

---

## 4. GPU Memory & Batch‐Sizing

**A100 40 GiB, Llama‐2‐7B (7e9 params, FP16)**

1. **Model Weights**: 7e9 × 2 bytes ≈ 14 GiB.
2. **Overhead**: \~2 GiB for CUDA, activations → **25 GiB** left.

**KV‑cache size per token**:

```
2 (key+value) × 2 bytes × 4096 hidden × 32 layers ≈ 0.5 MiB/token
```

**Max batched tokens**:

```
25 GiB / 0.5 MiB ≈ 51 200 tokens
→ use ~80% safety → 40 960 tokens
```

(`--max-num-batched-tokens 40960`)

**Max sequences** (max concurrent requests):

```
40960 tokens / 256 tokens/request ≈ 160
→ round to 128 or 160 (`--max-num-seqs`)
```

**Block‐size**:

- Controls streaming granularity.
- Smaller (8–16) → lower inter-token latency.
- Larger (32–64) → higher throughput.
- Start at **16**.

---

## Experiment Metric Calculations

For each batch of N concurrent requests, collect these before/after counters and derive:

- Energy Delta (J)

   - E0 = energy counter at start

   - E1 = energy counter at end

   - EnergyBatch = E1 - E0

- Time Delta (s)

   - t0 = perf_counter() at start

   - t1 = perf_counter() at end

   - TimeBatch = t1 - t0

- Token Counts

   - PromptDelta = v1.prompt_tokens_total - v0.prompt_tokens_total

   - GenDelta    = v1.generation_tokens_total - v0.generation_tokens_total

   - TotalDelta  = PromptDelta + GenDelta

- Joules per Token

   - JoulesPerToken = EnergyBatch / TotalDelta -> Joules
   - PowerPerBatch = EnergyBatch/ TimeBatch -> Watts (J/s) (average power draw during batch processing, how many joules were used each second, on average, during that request)

- Stage Durations (s)

   - PrefillTime   = v1.request_prefill_time_sum   - v0.request_prefill_time_sum

   - InferenceTime= v1.request_inference_time_sum - v0.request_inference_time_sum

   - DecodeTime   = v1.request_decode_time_sum    - v0.request_decode_time_sum

- Average Latencies (s)

   - E2E Mean     = (v1.e2e_latency_sum   - v0.e2e_latency_sum)   / (v1.e2e_latency_count   - v0.e2e_latency_count)

   - TTFT Mean    = (v1.time_to_first_sum - v0.time_to_first_sum) / (v1.time_to_first_count - v0.time_to_first_count)

   - Token Mean   = (v1.time_per_token_sum- v0.time_per_token_sum)/ (v1.time_per_token_count- v0.time_per_token_count)

- Per-Request Averages

   - PrefillAvg   = PrefillTime   / N

   - InferenceAvg= InferenceTime / N

   - DecodeAvg   = DecodeTime    / N

- Energy Breakdown by Stage

   - JoulesPrefill   = EnergyBatch × (PrefillTime   / TimeBatch)

   - JoulesInference= EnergyBatch × (InferenceTime / TimeBatch)

   - JoulesDecode    = EnergyBatch × (DecodeTime    / TimeBatch)

- Energy per Token by Stage

   - J_perPromptToken = JoulesPrefill   / PromptDelta

   - J_perGenToken    = JoulesInference / GenDelta

   - J_perTotalToken  = EnergyBatch      / TotalDelta
- Optional? Percentiles (not sure)

   - Fetch histogram buckets (_bucket, _count, _sum) and scan to compute P50/P95/P99 for TTFT or E2E.

### Why the division happens with latency_s (wall clocl time on client side) and not e2e latency?

####  latency_s –  end-to-end wall-clock

t0 = time.perf_counter()

 ─── send batch ───▶

│     … network, server work, streaming back, client post-proc  
 ◀──────────────────  
t1 = time.perf_counter()

latency_s = t1 - t0

Starts when the client issues the HTTP/gRPC call for the batch.

Ends when the client has fully received all responses and returned from await process_batch(...).

Includes:

- Network round-trip time (client → server, server → client)

- vLLM scheduling + compute (prefill, inference, decode)

- Any client-side parsing or post-processing (e.g. splitting, JSON parsing)

#### e2e latency

Measured entirely inside the vLLM server process.

Starts when vLLM begins handling each individual request (right after it’s dequeued from its internal queue).

Ends when vLLM has finished generating and streaming that request’s tokens.

Includes all three pipeline stages:

   - Prefill: processing the prompt through the model once.

   - Inference: iterative forward passes for new tokens.

   - Decode: converting logits to actual tokens/strings.

Excludes any time spent:

   - In client’s network stack before the request reaches the server.

   - In client’s Python code after the last token arrives.

   - (Usually) any scheduling delay before the request enters vLLM’s queue—unless vLLM’s own queueing is instrumented.

# Prompt Engineering