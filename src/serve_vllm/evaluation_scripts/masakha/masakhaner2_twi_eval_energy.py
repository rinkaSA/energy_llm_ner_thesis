import os
import re
import json
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


# Updated system message for Twi language
system_msg = {
    "role": "system",
    "content": (
        "You are a Named Entity Recognition (NER) engine for Twi language.  \n"
        "- **Input:** a sentence in **Twi** language.  \n"
        "- **Output:** **only** a **single**, **valid** JSON object with four arrays: `PER`, `ORG`, `LOC`, `DATE`.  \n"
        "- `PER` for person names, `ORG` for organizations, `LOC` for locations, and `DATE` for dates and time expressions.  \n"
        "- Do **not** output any extra text, bulleted lists, or explanation.  \n"
        "- JSON must be parseable by `json.loads`."
    )
}
# Examples in Twi with English translations as comments
examples = [
    {"role":"user", "content":"Sentence: Kofi Annan kɔɔ United Nations dwumadibea wɔ New York."},
    # Translation: "Kofi Annan went to the United Nations headquarters in New York."
    {"role":"assistant","content":(
        '{\n'
        '  "PER": ["Kofi Annan"],\n'
        '  "ORG": ["United Nations"],\n'
        '  "LOC": ["New York"],\n'
        '  "DATE": []\n'
        '}'
    )}
]

def make_messages_for(sentence: str):
    return [system_msg] + examples + [
        {"role":"user", "content":f"Sentence: {sentence}"}
    ]


ENERGY_METRIC_NAME = "DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION"  
ARTIFACTS_DIR = "INFERENCE_TWINER_1-128_awq_mistrall_full"


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
            obj = loader(block)
            if isinstance(obj, dict):
                parsed = obj
                break
        except Exception:
            continue

    # 7) Normalize keys→upper, values→list[str]
    result = {}
    for k, v in parsed.items():
        key = k.strip().upper()
        if isinstance(v, str):
            items = [x.strip() for x in v.split(",") if x.strip()]
        elif isinstance(v, (list, tuple)):
            items = [str(x).strip() for x in v if str(x).strip()]
        else:
            items = []
        result[key] = items

    return result


def read_energy_joules() -> float:
    """Scrape the DCGM exporters gauge from localhost:9400/metrics."""
    r = requests.get(ENERGY_URL, timeout=1.0).text
    for line in r.splitlines():
        if line.startswith(ENERGY_METRIC_NAME):
            return float(line.split()[-1])  # already in millijoules
    raise RuntimeError(f"{ENERGY_METRIC_NAME} not found in /metrics")


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
            return result["choices"][0]["message"]["content"]
    
    tasks = [single_request(s) for s in sentences]
    return await asyncio.gather(*tasks)


async def process_batch(session, prompts, max_tokens):
    """Your original fan-out of one-prompt → one HTTP call."""
    async def single_request(prompt):
        payload = {
            "model": "/model",
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0,
        }
        async with session.post(SERVER_URL, json=payload) as resp:
            result = await resp.json()
            return result["choices"][0]["text"]

    tasks = [single_request(p) for p in prompts]
    return await asyncio.gather(*tasks)


def numpy_serializer(obj):
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    # let the JSON encoder handle other types or raise
    raise TypeError(f"Type {obj.__class__.__name__} not serializable")


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
        raise RuntimeError(f"Missing vLLM metrics: {missing}")
    return m


async def process_and_measure(session, prompts, max_tokens):
    e0 = read_energy_joules()
    v0 = read_vllm_metrics()
    t0 = time.perf_counter()

    responses = await process_batch_chat(session, prompts, max_tokens)

    t1 = time.perf_counter()
    e1 = read_energy_joules()
    v1 = read_vllm_metrics()

    joules = (e1 - e0) / 1000  # IN MILIJOULES ->> convert to Joules
    latency = t1 - t0

    prompt_t = v1["prompt_tokens_total"] - v0["prompt_tokens_total"]
    gen_t = v1["generation_tokens_total"] - v0["generation_tokens_total"]
    total_t = prompt_t + gen_t

    e2e_latency_mean = (v1["e2e_latency_sum"] - v0["e2e_latency_sum"]) / \
                       (v1["e2e_latency_count"] - v0["e2e_latency_count"])  # per request latencies/ number of requests
    ttft_mean = (v1["time_to_first_sum"] - v0["time_to_first_sum"]) / \
                (v1["time_to_first_count"] - v0["time_to_first_count"])
    time_per_token_mean = (v1["time_per_token_sum"] - v0["time_per_token_sum"]) / \
                         (v1["time_per_token_count"] - v0["time_per_token_count"])

    prefill_total = v1["request_prefill_time_sum"] - v0["request_prefill_time_sum"]
    inference_total = v1["request_inference_time_sum"] - v0["request_inference_time_sum"]
    decode_total = v1["request_decode_time_sum"] - v0["request_decode_time_sum"]

    # assume energy ~ time (gpu and cpu draw roughly constant over the run
    joules_prefill = joules * (prefill_total / e2e_latency_mean)  # CHANGED FROM latency!
    joules_inference = joules * (inference_total / e2e_latency_mean)
    joules_decode = joules * (decode_total / e2e_latency_mean)

    diff = latency - e2e_latency_mean

    telemetry = {
        "batch_size": len(prompts),
        "latency_s": latency,
        "energy_j": joules,
        "prompt_tokens": prompt_t,
        "generation_tokens": gen_t,
        "total_tokens": total_t,

        "e2e_latency_mean": e2e_latency_mean,  # COMPARE THIS TO LATENCY_S!
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

        "J_prefill_per_prompt_token": joules_prefill / prompt_t if prompt_t > 0 else 0,
        "J_inf_per_gen_token": joules_inference / gen_t if gen_t > 0 else 0,
        "J_per_total_token": joules / total_t if total_t > 0 else 0,
        
        "J_total_per_prompt_token": joules / prompt_t if prompt_t > 0 else 0,
        "J_total_per_gen_token": joules / gen_t if gen_t > 0 else 0,
        "J_total_per_token": joules / total_t if total_t > 0 else 0,

        "avg_power_draw": joules / e2e_latency_mean  # in Watts
    }
    return responses, telemetry


def get_bio_tags(sentence, entities):
    """
    Convert a sentence and its extracted entity mentions (dictionary) into token-level BIO tags.
    
    Parameters:
      sentence (str): The input sentence.
      entities (dict): Mapping of entity types to a list of mention strings.
    
    Returns:
      tokens (list): List of tokens (using whitespace split).
      tags (list): List of BIO tags in the same order as tokens.
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


async def evaluate_ner_pipeline_masakhaner2_twi(
    test_dataset, label_list, batch_size, max_new_tokens=150
):
    all_gold, all_pred = [], []
    generated_results = []
    batch_telemetry = []  

    async with aiohttp.ClientSession() as session:
        for i in tqdm(range(0, len(test_dataset), batch_size)):
            batch = test_dataset.select(range(i, min(i+batch_size, len(test_dataset)))).to_list()
            sents = [" ".join(ex["tokens"]) for ex in batch]
            golds = [[label_list[t] for t in ex["ner_tags"]] for ex in batch]

            try:
                responses, telem = await process_and_measure(session, sents, max_new_tokens)
                batch_telemetry.append(telem)

                for sent, resp, g in zip(sents, responses, golds):
                    text = resp.strip()
                    generated_results.append({"prompt": sent, "generated_response": text})

                    ents = parse_response_new(text)
                    _, pred_tags = get_bio_tags(sent, ents)
                    
                    # Ensure prediction list has the same length as gold
                    if len(pred_tags) < len(g):
                        pred_tags += ["O"] * (len(g) - len(pred_tags))
                    elif len(pred_tags) > len(g):
                        pred_tags = pred_tags[:len(g)]
                        
                    all_gold.append(g)
                    all_pred.append(pred_tags)

            except Exception as e:
                print(f"Batch {i//batch_size} failed:", e)
                continue

    # compute overall NER metrics
    ner_metrics = {
        "precision": precision_score(all_gold, all_pred),
        "recall": recall_score(all_gold, all_pred),
        "f1": f1_score(all_gold, all_pred),
        "cls_report": classification_report(all_gold, all_pred, output_dict=True)
    }

    return ner_metrics, generated_results, batch_telemetry


async def main():
    # Load MasakhaNER2 Twi dataset
    test_ds = load_dataset("masakhane/masakhaner2", "twi", split="test")
    labels = test_ds.features["ner_tags"].feature.names

    print(f"Loaded MasakhaNER2 Twi dataset with {len(test_ds)} samples and {len(labels)} labels: {labels}")

    test_subset = test_ds  # Use the full test set for evaluation
    # Select a subset for testing if needed
    #test_subset = test_ds.select(range(min(320, len(test_ds))))
    print(f"Testing with {len(test_subset)} samples")

    for B in [1, 4, 8, 16, 32, 64, 128]:
        print(f"Processing with batch size {B}...")
        ner_metrics, gen_responses, batch_telemetry = await evaluate_ner_pipeline_masakhaner2_twi(
            test_subset, labels, batch_size=B
        )

        # Save generated responses
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

        # Save telemetry data
        telem_folder = os.path.join(ARTIFACTS_DIR, "telemetry")
        os.makedirs(telem_folder, exist_ok=True)
        
        telemetry_path = os.path.join(telem_folder, f"telemetry_B_{B}.json")
        with open(telemetry_path, "w") as f:
            json.dump(batch_telemetry, f, indent=2, default=numpy_serializer)

        # Calculate mean metrics across batches
        mean_metrics = {
            "avg_latency_s": np.mean([t["latency_s"] for t in batch_telemetry]),
            "avg_energy_j": np.mean([t["energy_j"] for t in batch_telemetry]),
            "avg_joules_per_token": np.mean([t["J_total_per_token"] for t in batch_telemetry]),

            "mean_e2e_latency_s": np.mean([t["e2e_latency_mean"] for t in batch_telemetry]),
            "mean_ttfb_s": np.mean([t["ttft_mean"] for t in batch_telemetry]),
            "mean_time_per_token_s": np.mean([t["time_per_token_mean"] for t in batch_telemetry]),
            "mean_diff": np.mean([t["diff"] for t in batch_telemetry]),

            "mean_prefill_total_s": np.mean([t["prefill_total_s"] for t in batch_telemetry]),
            "mean_inference_total_s": np.mean([t["inference_total_s"] for t in batch_telemetry]),
            "mean_decode_total_s": np.mean([t["decode_total_s"] for t in batch_telemetry]),

            "mean_prefill_avg_s": np.mean([t["prefill_avg_s"] for t in batch_telemetry]),
            "mean_inference_avg_s": np.mean([t["inference_avg_s"] for t in batch_telemetry]),
            "mean_decode_avg_s": np.mean([t["decode_avg_s"] for t in batch_telemetry]),

            "mean_joules_prefill": np.mean([t["joules_prefill"] for t in batch_telemetry]),
            "mean_joules_inference": np.mean([t["joules_inference"] for t in batch_telemetry]),
            "mean_joules_decode": np.mean([t["joules_decode"] for t in batch_telemetry]),

            "mean_J_prefill_per_prompt_token": np.mean([t["J_prefill_per_prompt_token"] for t in batch_telemetry]),
            "mean_J_inf_per_gen_token": np.mean([t["J_inf_per_gen_token"] for t in batch_telemetry]),
            "mean_J_per_total_token": np.mean([t["J_per_total_token"] for t in batch_telemetry]),
            "mean_J_total_per_prompt_token": np.mean([t["J_total_per_prompt_token"] for t in batch_telemetry]),
            
            "mean_J_total_per_gen_token": np.mean([t["J_total_per_gen_token"] for t in batch_telemetry]),
            "mean_J_total_per_token": np.mean([t["J_total_per_token"] for t in batch_telemetry]),
            "mean_avg_power_draw_W": np.mean([t["avg_power_draw"] for t in batch_telemetry]),
            
            # Add NER performance metrics
            "ner_precision": ner_metrics["precision"],
            "ner_recall": ner_metrics["recall"],
            "ner_f1": ner_metrics["f1"]
        }
        
        mean_metrics_path = os.path.join(ARTIFACTS_DIR, f"mean_metrics_B_{B}.json")
        with open(mean_metrics_path, "w") as f:
            json.dump(mean_metrics, f, indent=2)
        
        print(f"Batch size {B} completed: F1={ner_metrics['f1']:.4f}, Latency={mean_metrics['avg_latency_s']:.4f}s, Energy={mean_metrics['avg_energy_j']:.4f}J")


if __name__ == "__main__":
    asyncio.run(main())
