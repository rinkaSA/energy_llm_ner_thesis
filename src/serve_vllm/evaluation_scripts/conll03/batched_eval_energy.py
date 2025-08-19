import os
import re
import time
import json
import asyncio
import aiohttp
import requests
import numpy as np
import mlflow
from tqdm import tqdm
from datasets import load_dataset
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report


from dotenv import load_dotenv
load_dotenv()


SERVER_URL = os.getenv("SERVER_URL")
ENERGY_URL = os.getenv("ENERGY_URL")
VLLM_METRICS_URL =os.getenv("VLLM_METRICS_URL")

MLFLOW_TRACKING_URI   = os.getenv("MLFLOW_TRACKING_URI")

INSTRUCTION = """### Instruction:
You are an expert in natural language processing annotation. 
Given a sentence, identify and classify each named entity into one of the following types: 
LOC (Location), MISC (Miscellaneous), ORG (Organization), or PER (Person).

For example, consider the sentence:
'Brazilian Planning Minister Antonio Kandir will submit to a draft copy of the 1997 federal budget to Congress on Thursday, a ministry spokeswoman said.'

Expected output: {'MISC': ['Brazilian'], 'PER': ['Antonio Kandir'], 'ORG': ['Congress']}

Given the sentence below perform a task and include in the resulting output in json style format as in example only."""

ENERGY_METRIC_NAME    = "DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION"  
ARTIFACTS_DIR = "inference_artifacts_1_128_awq_mistral"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

def read_energy_joules() -> float:
    """Scrape the DCGM exporters gauge from localhost:9400/metrics."""
    r = requests.get(ENERGY_URL, timeout=1.0).text
    for line in r.splitlines():
        if line.startswith(ENERGY_METRIC_NAME):
            return float(line.split()[-1])
    raise RuntimeError(f"{ENERGY_METRIC_NAME} not found in /metrics")

async def process_batch(session, prompts, max_tokens):
    """Your original fan-out of one-prompt → one HTTP call."""
    async def single_request(prompt):
        payload = {
            "model": "/model",
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0
        }
        async with session.post(SERVER_URL, json=payload) as resp:
            resp.raise_for_status()
            j = await resp.json()
            return j["choices"][0]["text"]

    tasks = [single_request(p) for p in prompts]
    return await asyncio.gather(*tasks)



def read_vllm_metrics() -> dict:
    """
    Scrape vLLM's Prometheus /metrics and return the current cumulative counters:
      - prompt_tokens_total
      - generation_tokens_total
      - e2e_request_latency_seconds_sum & _count
      - time_to_first_token_seconds_sum & _count
      - time_per_output_token_seconds_sum & _count
    """
    text = requests.get(VLLM_METRICS_URL).text
    m = {
        "prompt_tokens_total":         None,
        "generation_tokens_total":     None,
        "e2e_latency_sum":             None,
        "e2e_latency_count":           None,
        "time_to_first_sum":           None,
        "time_to_first_count":         None,
        "time_per_token_sum":          None,
        "time_per_token_count":        None,
        "request_prefill_time_sum":    None,
        "request_inference_time_sum":  None,
        "request_decode_time_sum":     None
    }
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        parts = line.split()
        if len(parts) < 2:
            continue

        raw_name, raw_val = parts[0], parts[-1]
        name = raw_name.split("{", 1)[0]

        try:
            val = float(raw_val)
        except ValueError:
            continue

        if name == "vllm:prompt_tokens_total":
            m["prompt_tokens_total"] = val
        elif name == "vllm:generation_tokens_total":
            m["generation_tokens_total"] = val

        # THESE VALUES ARE NOT EXPOSED IF NO REQUESTS AT ALL WERE MADE! run smth before
        elif name.startswith( "vllm:e2e_request_latency_seconds_sum"):
            m["e2e_latency_sum"] = val
        elif name == "vllm:e2e_request_latency_seconds_count":
            m["e2e_latency_count"] = val

        elif name == "vllm:time_to_first_token_seconds_sum":
            m["time_to_first_sum"] = val
        elif name == "vllm:time_to_first_token_seconds_count":
            m["time_to_first_count"] = val

        elif name == "vllm:time_per_output_token_seconds_sum":
            m["time_per_token_sum"] = val
        elif name == "vllm:time_per_output_token_seconds_count":
            m["time_per_token_count"] = val
        elif name == "vllm:request_prefill_time_seconds_sum":
            m["request_prefill_time_sum"] = val
        elif name == "vllm:request_inference_time_seconds_sum":
            m["request_inference_time_sum"] = val
        elif name == "vllm:request_decode_time_seconds_sum":
            m["request_decode_time_sum"] = val   

    missing = [k for k, v in m.items() if v is None]
    if missing:
        raise RuntimeError(f"Missing vLLM metrics: {missing}")
    return m

async def process_and_measure(session, prompts, max_tokens):
    e0    = read_energy_joules()
    v0    = read_vllm_metrics()
    t0    = time.perf_counter()

    responses = await process_batch(session, prompts, max_tokens)

    t1    = time.perf_counter()
    e1    = read_energy_joules()
    v1    = read_vllm_metrics()

    joules    = (e1 - e0) / 1000 # IN MILIJOULES ->> convert to Joules
    latency   = t1 - t0

    prompt_t  = v1["prompt_tokens_total"]     - v0["prompt_tokens_total"]
    gen_t     = v1["generation_tokens_total"] - v0["generation_tokens_total"]
    total_t   = prompt_t + gen_t

    e2e_latency_mean      = (v1["e2e_latency_sum"]      - v0["e2e_latency_sum"])      \
                             / (v1["e2e_latency_count"] - v0["e2e_latency_count"])
    ttft_mean             = (v1["time_to_first_sum"]    - v0["time_to_first_sum"])    \
                             / (v1["time_to_first_count"] - v0["time_to_first_count"])
    time_per_token_mean   = (v1["time_per_token_sum"]   - v0["time_per_token_sum"])   \
                             / (v1["time_per_token_count"] - v0["time_per_token_count"])

    prefill_total    = v1["request_prefill_time_sum"]-v0["request_prefill_time_sum"]
    inference_total  = v1["request_inference_time_sum"]-v0["request_inference_time_sum"]
    decode_total     = v1["request_decode_time_sum"]-v0["request_decode_time_sum"]

    # assume energy ~ time (gpu and cpu draw roughly constant over the run
    joules_prefill    = joules * (prefill_total   / latency)
    joules_inference  = joules * (inference_total / latency)
    joules_decode     = joules * (decode_total    / latency)

    diff = latency - e2e_latency_mean

    telemetry = {
        "batch_size":           len(prompts),
        "latency_s":            latency,
        "energy_j":             joules,
        "prompt_tokens":        prompt_t,
        "generation_tokens":    gen_t,
        "total_tokens":         total_t,

        "e2e_latency_mean":     e2e_latency_mean, # COMPARE THIS TO LATENSY_S!
        "ttft_mean":            ttft_mean,
        "time_per_token_mean":  time_per_token_mean,

        "diff" : diff,
    
        "prefill_total_s":   prefill_total,
        "inference_total_s": inference_total,
        "decode_total_s":    decode_total,

        "prefill_avg_s":     prefill_total   / len(prompts),
        "inference_avg_s":   inference_total / len(prompts),
        "decode_avg_s":      decode_total    / len(prompts),

        "joules_prefill":      joules_prefill,
        "joules_inference":    joules_inference,
        "joules_decode":       joules_decode,

        "J_prefill_per_prompt_token":  joules_prefill   / prompt_t,
        "J_inf_per_gen_token":     joules_inference / gen_t,
        "J_per_total_token":   joules          / total_t, # SHOULD I ADD PERCENTILES FOR TTTF AND E2E
        
        "J_total_per_prompt_token":     joules / prompt_t,
        "J_total_per_gen_token":        joules / gen_t,
        "J_total_per_token":            joules / total_t,

        "avg_power_draw": joules / latency # in Watts
    }
    return responses, telemetry


async def evaluate_ner_pipeline_conll03(
    test_dataset, label_list, batch_size, max_new_tokens=150
):
    all_gold, all_pred = [], []
    generated_results  = []
    batch_telemetry    = []  

    async with aiohttp.ClientSession() as session:
        for i in tqdm(range(0, len(test_dataset), batch_size)):
            batch = test_dataset.select(range(i, min(i+batch_size, len(test_dataset)))).to_list()
            sents = [" ".join(ex["tokens"]) for ex in batch]
            golds = [[label_list[t] for t in ex["ner_tags"]] for ex in batch]
            prompts = [f"{INSTRUCTION}\nNow, given the sentence: {s} ### Response:"
                       for s in sents]

            try:
                responses, telem = await process_and_measure(session, prompts, max_new_tokens)
                batch_telemetry.append(telem)

                for sent, resp, g in zip(sents, responses, golds):
                    # strip off any header before the JSON
                    text = resp.split("### Response:")[-1].strip()
                    generated_results.append({"prompt": sent, "generated_response": text})

                    ents = parse_response(text)
                    _, pred_tags = get_bio_tags(sent, ents)
                    all_gold.append(g)
                    all_pred.append(pred_tags)

            except Exception as e:
                print(f"Batch {i//batch_size} failed:", e)
                continue

    # compute overall NER metrics
    ner_metrics = {
        "precision": precision_score(all_gold, all_pred),
        "recall":    recall_score(all_gold, all_pred),
        "f1":        f1_score(all_gold, all_pred),
        "cls_report": classification_report(all_gold, all_pred, output_dict=True)
    }

    return ner_metrics, generated_results, batch_telemetry

async def main():
    experiment_name = "Server_Evaluation"
    run_name = "batched_energy"
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment_name)
    test_ds = load_dataset("conll2003", trust_remote_code=True, split="test").select(range(10*32)) # make it devidable to the batch size!
    labels  = test_ds.features["ner_tags"].feature.names

    for B in [1, 4, 8, 16, 32, 64, 128]: # [1, 4, 8, 16, 32, 64, 128]:
        run_name = f"LLama2_batch{B}"
        with mlflow.start_run(run_name=run_name, log_system_metrics=True) as run:
            mlflow.log_param("batch_size", B)
            mlflow.log_param("prompt", INSTRUCTION)

            ner_metrics, gen_responses, batch_telemetry = await evaluate_ner_pipeline_conll03(
                test_ds, labels, batch_size=B
            )

            # Log NER metrics
            mlflow.log_metric("precision", ner_metrics["precision"])
            mlflow.log_metric("recall",    ner_metrics["recall"])
            mlflow.log_metric("f1",        ner_metrics["f1"])
            for ent, scores in ner_metrics["cls_report"].items():
                if isinstance(scores, dict):
                    for name, val in scores.items():
                        key = re.sub(r'[^a-zA-Z0-9_]', '_', f"{ent}_{name}")
                        mlflow.log_metric(key, val)
            for step, telem in enumerate(batch_telemetry):
                mlflow.log_metrics({
                    "latency_s":          telem["latency_s"],
                    "energy_j":           telem["energy_j"],
                    "prompt_tokens":      telem["prompt_tokens"],
                    "generation_tokens":  telem["generation_tokens"],
                    "total_tokens":       telem["total_tokens"],
                     "e2e_latency_mean":   telem["e2e_latency_mean"],
                    "ttft_mean":       telem["ttft_mean"],
                    "time_per_token_mean":  telem["time_per_token_mean"],
                    "diff":      telem["diff"],

                    "prefill_total_s":  telem["prefill_total_s"],
                    "inference_total_s": telem["inference_total_s"],
                    "decode_total_s":  telem["decode_total_s"],

                    "prefill_avg_s":   telem["prefill_avg_s"],
                    "inference_avg_s":  telem["inference_avg_s"],
                    "decode_avg_s":  telem["decode_avg_s"],
                    
                    "joules_prefill":    telem["joules_prefill"],
                    "joules_inference": telem["joules_inference"],
                    "joules_decode":      telem["joules_decode"],

                    "J_prefill_per_prompt_token": telem["J_prefill_per_prompt_token"],
                    "J_inf_per_gen_token": telem["J_inf_per_gen_token"],
                    "J_per_total_token": telem["J_per_total_token"],
                    "J_total_per_prompt_token": telem["J_total_per_prompt_token"],
                    "J_total_per_gen_token": telem["J_total_per_gen_token"],
                    "J_total_per_token": telem["J_total_per_token"],
                    "avg_power_draw_W": telem["avg_power_draw"]
                                    }, step=step)

            with open(f"telemetry_batch_{B}.json", "w") as f:
                json.dump(batch_telemetry, f, indent=2)
            mlflow.log_artifact(f"telemetry_batch_{B}.json")   


            mean_metrics = {
                "avg_latency_s":            np.mean([t["latency_s"] for t in batch_telemetry]),
                "avg_energy_j":             np.mean([t["energy_j"] for t in batch_telemetry]),
                "avg_joules_per_token":     np.mean([t["J_total_per_token"] for t in batch_telemetry]),

                "mean_e2e_latency_s":       np.mean([t["e2e_latency_mean"]       for t in batch_telemetry]),
                "mean_ttfb_s":              np.mean([t["ttft_mean"]             for t in batch_telemetry]),
                "mean_time_per_token_s":    np.mean([t["time_per_token_mean"]    for t in batch_telemetry]),
                "mean_diff":                np.mean([t["diff"]                   for t in batch_telemetry]),

                "mean_prefill_total_s":     np.mean([t["prefill_total_s"]       for t in batch_telemetry]),
                "mean_inference_total_s":   np.mean([t["inference_total_s"]     for t in batch_telemetry]),
                "mean_decode_total_s":      np.mean([t["decode_total_s"]        for t in batch_telemetry]),

                "mean_prefill_avg_s":       np.mean([t["prefill_avg_s"]         for t in batch_telemetry]),
                "mean_inference_avg_s":     np.mean([t["inference_avg_s"]       for t in batch_telemetry]),
                "mean_decode_avg_s":        np.mean([t["decode_avg_s"]          for t in batch_telemetry]),

                "mean_joules_prefill":      np.mean([t["joules_prefill"]        for t in batch_telemetry]),
                "mean_joules_inference":    np.mean([t["joules_inference"]      for t in batch_telemetry]),
                "mean_joules_decode":       np.mean([t["joules_decode"]         for t in batch_telemetry]),

                "mean_J_prefill_per_prompt_token": np.mean([t["J_prefill_per_prompt_token"] for t in batch_telemetry]),
                "mean_J_inf_per_gen_token": np.mean([t["J_inf_per_gen_token"] for t in batch_telemetry]),
                "mean_J_per_total_token": np.mean([t["J_per_total_token"] for t in batch_telemetry]),
                "mean_J_total_per_prompt_token": np.mean([t["J_total_per_prompt_token"] for t in batch_telemetry]),
                
                "mean_J_total_per_gen_token": np.mean([t["J_total_per_gen_token"] for t in batch_telemetry]),
                "mean_J_total_per_token": np.mean([t["J_total_per_token"] for t in batch_telemetry]),
                "mean_avg_power_draw_W": np.mean([t["avg_power_draw"] for t in batch_telemetry])
            }
            mlflow.log_metrics(mean_metrics, step=step)

            responses_folder = os.path.join(ARTIFACTS_DIR, "server_response_results")
            os.makedirs(responses_folder, exist_ok=True)

            responses_path = os.path.join(responses_folder, f"responses_B{B}.json")
            with open(responses_path, "w") as f:
                json.dump(gen_responses, f, indent=2)

            mlflow.log_artifact(responses_path, artifact_path="server_response_results")

if __name__ == "__main__":
    from energy_ner_llm.src.serve_vllm.evaluation_scripts.eval_llm_batches import parse_response, get_bio_tags
    asyncio.run(main())