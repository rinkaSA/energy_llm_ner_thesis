import json
import os
import re
import ast
import time
import asyncio
import aiohttp
import requests
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
from dotenv import load_dotenv

load_dotenv()

SERVER_URL = os.getenv("SERVER_URL")
ENERGY_URL = os.getenv("ENERGY_URL")
VLLM_METRICS_URL = os.getenv("VLLM_METRICS_URL")

# System message for German NER task
system_msg = {
    "role": "system",
    "content": (
        "You are a German NER engine.\n"
        "- **Input:** a German sentence.\n"
        "- **Output:** **only** a **single**, **valid** JSON object with three arrays: `PER`, `ORG`, `LOC`.\n"
        "- Do **not** output any extra text, bulleted lists, or explanation.\n"
        "- JSON must be parseable by `json.loads`.\n"
        "- Extract named entities in German from the input sentence."
    )
}

# Few-shot examples for German NER
examples = [
    {
        "role": "user", 
        "content": "Sentence: Angela Merkel sprach bei den Vereinten Nationen in New York."
    },
    {
        "role": "assistant",
        "content": (
            '{\n'
            '  "PER": ["Angela Merkel"],\n'
            '  "ORG": ["Vereinten Nationen"],\n'
            '  "LOC": ["New York"]\n'
            '}'
        )
    }
]

def make_messages_for(sentence: str):
    return [system_msg] + examples + [
        {"role": "user", "content": f"Sentence: {sentence}"}
    ]

ENERGY_METRIC_NAME = "DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION"
ARTIFACTS_DIR = "INFERENCE_XTREME_GERMAN_1-128_awq_mistral"

os.makedirs(ARTIFACTS_DIR, exist_ok=True)

def parse_response_new(response_text: str) -> dict:
    """
    Robustly parse a JSON-like dict out of possibly-broken model output,
    e.g. missing the final '}' or containing single quotes.
    Returns a dict[str, list[str]] with upper-case keys.
    """
    # 1) Drop any leading index prefixes like "00 = '"
    cleaned = re.sub(r"^\s*\d+\s*=\s*", "", response_text, flags=re.MULTILINE)
    # 2) Strip wrapping quotes/newlines
    cleaned = cleaned.strip().strip("'\"")
    
    # 3) Find first '{' and walk forward counting braces
    start = cleaned.find("{")
    if start < 0:
        return {}
    depth = 0
    end_idx = None
    for i, ch in enumerate(cleaned[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end_idx = i
                break

    # 4) If we found a matching '}', grab that slice; otherwise append missing '}'s
    if end_idx is not None:
        block = cleaned[start : end_idx + 1]
    else:
        # "depth" is how many '{' were never closed
        block = cleaned[start:] + "}" * depth

    # 5) Remove any stray commas before the final brace, normalize quotes
    block = re.sub(r",\s*}", "}", block)
    block = block.replace("'", '"')

    # 6) Try parsing as JSON, then Python literal
    parsed = {}
    for loader in (json.loads, lambda s: ast.literal_eval(s)):
        try:
            parsed = loader(block)
            break
        except Exception:
            continue

    # 7) Normalize keys→upper, values→list[str]
    result = {}
    for k, v in parsed.items():
        key = k.strip().upper()
        if isinstance(v, str):
            items = [v.strip()] if v.strip() else []
        elif isinstance(v, (list, tuple)):
            items = [str(x).strip() for x in v if str(x).strip()]
        else:
            items = []
        result[key] = items

    return result

def get_bio_tags(sentence, entities):
    """
    Convert a sentence and its extracted entity mentions (dictionary) into token-level BIO tags.
    """
    tokens = sentence.split()
    tags = ["O"] * len(tokens)
    
    for ent_type, mentions in entities.items():
        for mention in mentions:
            mention_tokens = mention.split()
            for i in range(len(tokens) - len(mention_tokens) + 1):
                if tokens[i:i+len(mention_tokens)] == mention_tokens:
                    tags[i] = "B-" + ent_type
                    for j in range(1, len(mention_tokens)):
                        tags[i+j] = "I-" + ent_type
                    break  # Mark only the first occurrence
    return tokens, tags

def read_energy_joules() -> float:
    """Scrape the DCGM exporters gauge from localhost:9400/metrics."""
    try:
        r = requests.get(ENERGY_URL, timeout=1.0).text
        for line in r.splitlines():
            if line.startswith(ENERGY_METRIC_NAME):
                return float(line.split()[-1])
        raise RuntimeError(f"{ENERGY_METRIC_NAME} not found in /metrics")
    except Exception:
        return 0.0

async def process_batch_chat(session, sentences, max_tokens):
    async def single_request(sentence):
        messages = make_messages_for(sentence)
        payload = {
            "model": "/model",
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "stop": ["}"]   
        }
    
        async with session.post(SERVER_URL, json=payload) as resp:
            result = await resp.json()
            return result
    
    tasks = [single_request(s) for s in sentences]
    return await asyncio.gather(*tasks, return_exceptions=True)

def numpy_serializer(obj):
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    # let the JSON encoder handle other types or raise
    raise TypeError(f"Type {obj.__class__.__name__} not serializable")

def read_vllm_metrics() -> dict:
    """
    Scrape vLLM's Prometheus /metrics and return the current cumulative counters.
    """
    try:
        text = requests.get(VLLM_METRICS_URL).text
        m = {
            "prompt_tokens_total": None,
            "generation_tokens_total": None,
            "e2e_latency_sum": None,
            "e2e_latency_count": None,
            "time_to_first_sum": None,
            "time_to_first_count": None,
            "time_per_token_sum": None,
            "time_per_token_count": None,
            "request_prefill_time_sum": None,
            "request_inference_time_sum": None,
            "request_decode_time_sum": None
        }
        
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) < 2:
                continue

            raw_name, raw_val = parts[0], parts[1]
            name = raw_name.split("{", 1)[0]

            try:
                val = float(raw_val)
            except ValueError:
                continue

            if name == "vllm:prompt_tokens_total":
                m["prompt_tokens_total"] = val
            elif name == "vllm:generation_tokens_total":
                m["generation_tokens_total"] = val
            elif name.startswith("vllm:e2e_request_latency_seconds_sum"):
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
            print(f"Warning: Missing vLLM metrics: {missing}")
            # Return zeros for missing metrics
            for k in missing:
                m[k] = 0.0
        return m
    except Exception as e:
        print(f"Error reading vLLM metrics: {e}")
        return {k: 0.0 for k in [
            "prompt_tokens_total", "generation_tokens_total", "e2e_latency_sum",
            "e2e_latency_count", "time_to_first_sum", "time_to_first_count",
            "time_per_token_sum", "time_per_token_count", "request_prefill_time_sum",
            "request_inference_time_sum", "request_decode_time_sum"
        ]}

async def process_and_measure(session, prompts, max_tokens):
    e0 = read_energy_joules()
    v0 = read_vllm_metrics()
    t0 = time.perf_counter()

    responses = await process_batch_chat(session, prompts, max_tokens)

    t1 = time.perf_counter()
    e1 = read_energy_joules()
    v1 = read_vllm_metrics()

    joules = (e1 - e0) / 1000  # Convert millijoules to joules
    latency = t1 - t0

    prompt_t = v1["prompt_tokens_total"] - v0["prompt_tokens_total"]
    gen_t = v1["generation_tokens_total"] - v0["generation_tokens_total"]
    total_t = prompt_t + gen_t

    # Avoid division by zero
    e2e_count_diff = v1["e2e_latency_count"] - v0["e2e_latency_count"]
    ttft_count_diff = v1["time_to_first_count"] - v0["time_to_first_count"]
    tpt_count_diff = v1["time_per_token_count"] - v0["time_per_token_count"]

    e2e_latency_mean = (v1["e2e_latency_sum"] - v0["e2e_latency_sum"]) / max(e2e_count_diff, 1)
    ttft_mean = (v1["time_to_first_sum"] - v0["time_to_first_sum"]) / max(ttft_count_diff, 1)
    time_per_token_mean = (v1["time_per_token_sum"] - v0["time_per_token_sum"]) / max(tpt_count_diff, 1)

    prefill_total = v1["request_prefill_time_sum"] - v0["request_prefill_time_sum"]
    inference_total = v1["request_inference_time_sum"] - v0["request_inference_time_sum"]
    decode_total = v1["request_decode_time_sum"] - v0["request_decode_time_sum"]

    # Proportional energy allocation
    total_processing_time = max(e2e_latency_mean * len(prompts), 1)
    joules_prefill = joules * (prefill_total / total_processing_time)
    joules_inference = joules * (inference_total / total_processing_time)
    joules_decode = joules * (decode_total / total_processing_time)

    diff = latency - e2e_latency_mean

    telemetry = {
        "batch_size": len(prompts),
        "latency_s": latency,
        "energy_j": joules,
        "prompt_tokens": prompt_t,
        "generation_tokens": gen_t,
        "total_tokens": total_t,
        "e2e_latency_mean": e2e_latency_mean,
        "ttft_mean": ttft_mean,
        "time_per_token_mean": time_per_token_mean,
        "diff": diff,
        "prefill_total_s": prefill_total,
        "inference_total_s": inference_total,
        "decode_total_s": decode_total,
        "prefill_avg_s": prefill_total / len(prompts),
        "inference_avg_s": inference_total / len(prompts),
        "decode_avg_s": decode_total / len(prompts),
        "joules_prefill": joules_prefill,
        "joules_inference": joules_inference,
        "joules_decode": joules_decode,
        "J_prefill_per_prompt_token": joules_prefill / max(prompt_t, 1),
        "J_inf_per_gen_token": joules_inference / max(gen_t, 1),
        "J_per_total_token": joules / max(total_t, 1),
        "J_total_per_prompt_token": joules / max(prompt_t, 1),
        "J_total_per_gen_token": joules / max(gen_t, 1),
        "J_total_per_token": joules / max(total_t, 1),
        "avg_power_draw": joules / max(e2e_latency_mean, 0.001)
    }
    return responses, telemetry

async def evaluate_ner_pipeline_xtreme_german(
    test_dataset, label_list, batch_size, max_new_tokens=150
):
    all_gold, all_pred = [], []
    generated_results = []
    batch_telemetry = []

    async with aiohttp.ClientSession() as session:
        for i in tqdm(range(0, len(test_dataset), batch_size)):
            batch = test_dataset[i:i+batch_size]
            sentences = []
            gold_batch = []
            
            for example in batch:
                # XTREME dataset structure: tokens and ner_tags
                tokens = example["tokens"]
                ner_tags = example["ner_tags"]
                
                sentence = " ".join(tokens)
                gold_tags = [label_list[tag] for tag in ner_tags]
                
                sentences.append(sentence)
                gold_batch.append(gold_tags)
            
            try:
                responses, telemetry = await process_and_measure(session, sentences, max_new_tokens)
                batch_telemetry.append(telemetry)
                
                for j, (response, gold_tags, sentence) in enumerate(zip(responses, gold_batch, sentences)):
                    generated_results.append({
                        "sentence": sentence,
                        "response": response,
                        "gold_tags": gold_tags
                    })
                    
                    # Handle exception responses
                    if isinstance(response, Exception):
                        print(f"Request failed for sentence {j}: {response}")
                        text = ""
                    # Extract response text
                    elif isinstance(response, dict) and "choices" in response and len(response["choices"]) > 0:
                        if "message" in response["choices"][0]:
                            text = response["choices"][0]["message"]["content"]
                        else:
                            text = response["choices"][0].get("text", str(response))
                    else:
                        text = str(response)
                    
                    # Parse entities and convert to BIO tags
                    entities = parse_response_new(text)
                    _, pred_tags = get_bio_tags(sentence, entities)
                    
                    # Ensure prediction list has the same length as gold
                    if len(pred_tags) < len(gold_tags):
                        pred_tags += ["O"] * (len(gold_tags) - len(pred_tags))
                    elif len(pred_tags) > len(gold_tags):
                        pred_tags = pred_tags[:len(gold_tags)]
                        
                    all_gold.append(gold_tags)
                    all_pred.append(pred_tags)

            except Exception as e:
                print(f"Batch {i//batch_size} failed:", e)
                # Add dummy predictions for failed batches to maintain alignment
                for gold_tags in gold_batch:
                    all_gold.append(gold_tags)
                    all_pred.append(["O"] * len(gold_tags))
                continue

    # Compute overall NER metrics
    if len(all_gold) > 0 and len(all_pred) > 0:
        ner_metrics = {
            "precision": precision_score(all_gold, all_pred),
            "recall": recall_score(all_gold, all_pred),
            "f1": f1_score(all_gold, all_pred),
            "cls_report": classification_report(all_gold, all_pred, output_dict=True)
        }
    else:
        ner_metrics = {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "cls_report": {}
        }

    return ner_metrics, generated_results, batch_telemetry

async def main():
    # Load XTREME German NER dataset
    test_ds = load_dataset("google/xtreme", "PAN-X.de", split="test", trust_remote_code=True)
    
    # Take only 1000 samples as requested
    test_subset = test_ds.select(range(min(1000, len(test_ds))))
    
    labels = test_ds.features["ner_tags"].feature.names
    print(f"Loaded XTREME German dataset with {len(test_subset)} samples and {len(labels)} labels: {labels}")

    for B in [1, 4, 8, 16, 32, 64, 128]:
        print(f"Processing with batch size {B}...")
        ner_metrics, gen_responses, batch_telemetry = await evaluate_ner_pipeline_xtreme_german(
            test_subset, labels, batch_size=B
        )

        # Save responses
        responses_folder = os.path.join(ARTIFACTS_DIR, "generated_responses")
        os.makedirs(responses_folder, exist_ok=True)
        responses_path = os.path.join(responses_folder, f"responses_B{B}.json")
        with open(responses_path, "w") as f:
            json.dump(gen_responses, f, indent=2, default=numpy_serializer)

        # Save NER metrics
        ner_folder = os.path.join(ARTIFACTS_DIR, "ner_metrics")
        os.makedirs(ner_folder, exist_ok=True)
        ner_metrics_path = os.path.join(ner_folder, f"ner_metrics_B_{B}.json")
        with open(ner_metrics_path, "w") as f:
            json.dump(ner_metrics, f, indent=2, default=numpy_serializer)

        # Save telemetry
        telem_folder = os.path.join(ARTIFACTS_DIR, "telemetry")
        os.makedirs(telem_folder, exist_ok=True)
        telemetry_path = os.path.join(telem_folder, f"telemetry_B_{B}.json")
        with open(telemetry_path, "w") as f:
            json.dump(batch_telemetry, f, indent=2, default=numpy_serializer)

        print(f"Batch size {B} completed. F1 score: {ner_metrics['f1']:.4f}")

        # Compute and save mean metrics
        if batch_telemetry:
            mean_metrics = {}
            for key in batch_telemetry[0].keys():
                if isinstance(batch_telemetry[0][key], (int, float)):
                    mean_metrics[key] = np.mean([t[key] for t in batch_telemetry])
            
            mean_metrics_path = os.path.join(ARTIFACTS_DIR, f"mean_metrics_B_{B}.json")
            with open(mean_metrics_path, "w") as f:
                json.dump(mean_metrics, f, indent=2, default=numpy_serializer)

    print("All batch sizes completed!")

if __name__ == "__main__":
    from evaluation_scripts.eval_llm_batches import parse_response, get_bio_tags
    asyncio.run(main())
