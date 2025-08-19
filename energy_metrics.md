
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


- CPU utilization - take the same stuff from cross-lingual-ner-with-multilinguah-sequence-translation

/utils/ resource_monitor.py + analyze_reesourcces.py + use it as wrapper to bash executable functions 
 relies on psutil - how we get the measureements - but we have 
