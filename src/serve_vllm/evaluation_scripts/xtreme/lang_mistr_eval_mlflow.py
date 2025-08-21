#!/usr/bin/env python3
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
import argparse
import mlflow

# ---------------------------
# Env & constants
# ---------------------------
load_dotenv()

SERVER_URL = os.getenv("SERVER_URL")
ENERGY_URL = os.getenv("ENERGY_URL")
VLLM_METRICS_URL = os.getenv("VLLM_METRICS_URL")
ENERGY_METRIC_NAME = "DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
FEW_SHOTS_PATH = "./prompts_in_all_languages/ner_few_shots_panx.json"

# ---------------------------
# Prompt template 
# ---------------------------

SYSTEM_TEMPLATE = (
    "You are a Named Entity Recognition (NER) annotator for the {language} language.\n"
    "- Input: a single {language} sentence (do not translate).\n"
    "- Output: only a single, valid JSON object with three arrays: `PER`, `ORG`, `LOC`.\n"
    "- Do not output any extra text, explanations, or markdown.\n"
    "- JSON must be parseable by json.loads.\n"
    "- Extract only entities that appear verbatim in the input sentence (no hallucinations).\n"
    "- Keep entity surface forms exactly as they appear; do not normalize or translate."
)

def full_lang_name(language: str) -> str:
    return {
        "de": "German",
        "en": "English",
        "ar": "Arabic",
        "bg": "Bulgarian",
        "zh": "Chinese"
    }.get(language, language)


def build_system_msg(language: str) -> dict:
    return {
        "role": "system",
        "content": SYSTEM_TEMPLATE.format(language=full_lang_name(language))
    }


def load_few_shots(language: str):
    with open(FEW_SHOTS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    if language not in data:
        raise ValueError(f"No few-shot examples found for language: {language}")
    return data[language]  

def make_messages_for(sentence: str, language: str, examples_for_lang: list):
    system_msg = build_system_msg(language)
    return [system_msg] + examples_for_lang + [
        {"role": "user", "content": f"Sentence: {sentence}"}
    ]
# ---------------------------
# Utils
# ---------------------------
def parse_response_new(response_text: str) -> dict:
    """
    Robustly parse a JSON-like dict out of possibly-broken model output,
    normalizing keys to upper-case.
    """
    cleaned = re.sub(r"^\s*\d+\s*=\s*", "", response_text, flags=re.MULTILINE)
    cleaned = cleaned.strip().strip("'\"")

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

    if end_idx is not None:
        block = cleaned[start:end_idx + 1]
    else:
        block = cleaned[start:] + "}" * depth

    block = re.sub(r",\s*}", "}", block)
    block = block.replace("'", '"')

    parsed = {}
    for loader in (json.loads, lambda s: ast.literal_eval(s)):
        try:
            parsed = loader(block)
            break
        except Exception:
            continue

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
                    break
    return tokens, tags

def numpy_serializer(obj):
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Type {obj.__class__.__name__} not serializable")

def read_energy_joules() -> float:
    """Read DCGM energy metric; returns joules (DCGM exposes millijoules)."""
    try:
        r = requests.get(ENERGY_URL, timeout=1.0).text
        for line in r.splitlines():
            if line.startswith(ENERGY_METRIC_NAME):
                return float(line.split()[-1]) / 1000.0
        raise RuntimeError(f"{ENERGY_METRIC_NAME} not found in /metrics")
    except Exception:
        return 0.0

def read_vllm_metrics() -> dict:
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

        for k, v in m.items():
            if v is None:
                m[k] = 0.0
        return m
    except Exception:
        return {k: 0.0 for k in [
            "prompt_tokens_total", "generation_tokens_total", "e2e_latency_sum",
            "e2e_latency_count", "time_to_first_sum", "time_to_first_count",
            "time_per_token_sum", "time_per_token_count", "request_prefill_time_sum",
            "request_inference_time_sum", "request_decode_time_sum"
        ]}

async def process_batch_chat(session, sentences, max_tokens, model_name, language):
    async def single_request(sentence):
        few_shots = load_few_shots(language)
        messages = make_messages_for(sentence, language, few_shots)
        payload = {
            "model": model_name,
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

async def process_and_measure(session, prompts, max_tokens, model_name, language):
    e0 = read_energy_joules()
    v0 = read_vllm_metrics()
    t0 = time.perf_counter()

    responses = await process_batch_chat(session, prompts, max_tokens, model_name, language)

    t1 = time.perf_counter()
    e1 = read_energy_joules()
    v1 = read_vllm_metrics()

    joules = max(e1 - e0, 0.0)
    latency = t1 - t0

    prompt_t = v1["prompt_tokens_total"] - v0["prompt_tokens_total"]
    gen_t = v1["generation_tokens_total"] - v0["generation_tokens_total"]
    total_t = prompt_t + gen_t

    e2e_count_diff = max(v1["e2e_latency_count"] - v0["e2e_latency_count"], 1)
    ttft_count_diff = max(v1["time_to_first_count"] - v0["time_to_first_count"], 1)
    tpt_count_diff = max(v1["time_per_token_count"] - v0["time_per_token_count"], 1)

    e2e_latency_mean = (v1["e2e_latency_sum"] - v0["e2e_latency_sum"]) / e2e_count_diff
    ttft_mean = (v1["time_to_first_sum"] - v0["time_to_first_sum"]) / ttft_count_diff
    time_per_token_mean = (v1["time_per_token_sum"] - v0["time_per_token_sum"]) / tpt_count_diff

    prefill_total = v1["request_prefill_time_sum"] - v0["request_prefill_time_sum"]
    inference_total = v1["request_inference_time_sum"] - v0["request_inference_time_sum"]
    decode_total = v1["request_decode_time_sum"] - v0["request_decode_time_sum"]

    total_processing_time = max(e2e_latency_mean * len(prompts), 1e-9)
    joules_prefill = joules * (prefill_total / total_processing_time) if total_processing_time > 0 else 0.0
    joules_inference = joules * (inference_total / total_processing_time) if total_processing_time > 0 else 0.0
    joules_decode = joules * (decode_total / total_processing_time) if total_processing_time > 0 else 0.0

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
        "prefill_total_s": prefill_total,
        "inference_total_s": inference_total,
        "decode_total_s": decode_total,
        "prefill_avg_s": prefill_total / max(len(prompts), 1),
        "inference_avg_s": inference_total / max(len(prompts), 1),
        "decode_avg_s": decode_total / max(len(prompts), 1),
        "joules_prefill": joules_prefill,
        "joules_inference": joules_inference,
        "joules_decode": joules_decode,
        "J_prefill_per_prompt_token": joules_prefill / max(prompt_t, 1),
        "J_inf_per_gen_token": joules_inference / max(gen_t, 1),
        "J_total_per_token": joules / max(total_t, 1),
        "J_total_per_prompt_token": joules / max(prompt_t, 1),
        "J_total_per_gen_token": joules / max(gen_t, 1),
        "avg_power_draw": joules / max(e2e_latency_mean, 1e-6),
    }
    return responses, telemetry

async def evaluate_ner_pipeline_xtreme(
    test_dataset, label_list, batch_size, model_name, max_new_tokens=150, language="de"
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
                tokens = example["tokens"]
                ner_tags = example["ner_tags"]
                sentence = " ".join(tokens)
                gold_tags = [label_list[tag] for tag in ner_tags]
                sentences.append(sentence)
                gold_batch.append(gold_tags)

            try:
                responses, telemetry = await process_and_measure(session, sentences, max_new_tokens, model_name, language)
                batch_telemetry.append(telemetry)

                for j, (response, gold_tags, sentence) in enumerate(zip(responses, gold_batch, sentences)):
                    generated_results.append({
                        "sentence": sentence,
                        "response": response,
                        "gold_tags": gold_tags
                    })

                    if isinstance(response, Exception):
                        text = ""
                    elif isinstance(response, dict) and "choices" in response and len(response["choices"]) > 0:
                        if "message" in response["choices"][0]:
                            text = response["choices"][0]["message"]["content"]
                        else:
                            text = response["choices"][0].get("text", str(response))
                    else:
                        text = str(response)

                    entities = parse_response_new(text)
                    _, pred_tags = get_bio_tags(sentence, entities)

                    if len(pred_tags) < len(gold_tags):
                        pred_tags += ["O"] * (len(gold_tags) - len(pred_tags))
                    elif len(pred_tags) > len(gold_tags):
                        pred_tags = pred_tags[:len(gold_tags)]

                    all_gold.append(gold_tags)
                    all_pred.append(pred_tags)

            except Exception as e:
                print(f"Batch {i//batch_size} failed:", e)
                for gold_tags in gold_batch:
                    all_gold.append(gold_tags)
                    all_pred.append(["O"] * len(gold_tags))
                continue

    if len(all_gold) > 0 and len(all_pred) > 0:
        ner_metrics = {
            "precision": precision_score(all_gold, all_pred),
            "recall": recall_score(all_gold, all_pred),
            "f1": f1_score(all_gold, all_pred),
            "cls_report": classification_report(all_gold, all_pred, output_dict=True)
        }
    else:
        ner_metrics = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "cls_report": {}}

    return ner_metrics, generated_results, batch_telemetry

# ---------------------------
# Main
# ---------------------------
def resolve_xtreme_subset(language: str) -> str:
    mapping = {
        "de": "PAN-X.de",
        "en": "PAN-X.en",
        "ar": "PAN-X.ar",
        "bg": "PAN-X.bg",
        "zh": "PAN-X.zh"
    }
    if language not in mapping:
        raise ValueError(f"Unsupported language: {language}")
    return mapping[language]

def build_artifacts_dir(language: str, batch_size: int, model_name: str):
    safe_model = re.sub(r"[^a-zA-Z0-9_.-]+", "_", model_name)
    return f"{language}_B{batch_size}_{safe_model}"

async def run(args):
    subset = resolve_xtreme_subset(args.language)
    print(f"Loading XTREME subset: {subset}")
    test_ds = load_dataset("google/xtreme", subset, split="test", trust_remote_code=True)
    test_subset = test_ds.select(range(min(1000, len(test_ds)))) 
    labels = test_ds.features["ner_tags"].feature.names
    print(f"Loaded {args.language} dataset with {len(test_subset)} samples and {len(labels)} labels: {labels}")

    artifacts_dir = build_artifacts_dir(args.language, args.batch_size, args.model)
    base_dir = "./inference_eval_artifacts/xtreme"
    full_artifacts_dir = os.path.join(base_dir, artifacts_dir)
    os.makedirs(full_artifacts_dir, exist_ok=True)

    experiment_name = f"{args.model}_xtreme"
    
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment_name)
    # ----- MLflow run -----
    with mlflow.start_run(run_name=f"xtreme_ner_{args.language}_B{args.batch_size}"):
        mlflow.set_tags({
            "language": args.language,
            "batch_size": str(args.batch_size),
            "model": args.model,
            "dataset": "xtreme",
            "n_samples" : str(len(test_subset))
        })
        mlflow.log_params({
            "language": args.language,
            "batch_size": args.batch_size,
            "model": args.model,
            "max_new_tokens": args.max_new_tokens,
            "dataset": "xtreme",
            "n_samples" : str(len(test_subset))
        })

        ner_metrics, gen_responses, batch_telemetry = await evaluate_ner_pipeline_xtreme(
            test_subset, labels, batch_size=args.batch_size, model_name=args.model,
            max_new_tokens=args.max_new_tokens, language=args.language
        )

        responses_folder = os.path.join(full_artifacts_dir, "generated_responses")
        os.makedirs(responses_folder, exist_ok=True)
        responses_path = os.path.join(responses_folder, f"responses_B{args.batch_size}.json")
        with open(responses_path, "w") as f:
            json.dump(gen_responses, f, indent=2, default=numpy_serializer)

        ner_folder = os.path.join(full_artifacts_dir, "ner_metrics")
        os.makedirs(ner_folder, exist_ok=True)
        ner_metrics_path = os.path.join(ner_folder, f"ner_metrics_B_{args.batch_size}.json")
        with open(ner_metrics_path, "w") as f:
            json.dump(ner_metrics, f, indent=2, default=numpy_serializer)

        telem_folder = os.path.join(full_artifacts_dir, "telemetry")
        os.makedirs(telem_folder, exist_ok=True)
        telemetry_path = os.path.join(telem_folder, f"telemetry_B_{args.batch_size}.json")
        with open(telemetry_path, "w") as f:
            json.dump(batch_telemetry, f, indent=2, default=numpy_serializer)

        # Log to MLflow
        mlflow.log_artifact(responses_path)
        mlflow.log_artifact(ner_metrics_path)
        mlflow.log_artifact(telemetry_path)
        # Also log metrics (flat)
        mlflow.log_metric("precision", float(ner_metrics.get("precision", 0.0)))
        mlflow.log_metric("recall", float(ner_metrics.get("recall", 0.0)))
        mlflow.log_metric("f1", float(ner_metrics.get("f1", 0.0)))

        if batch_telemetry:
            numeric_keys = [k for k, v in batch_telemetry[0].items() if isinstance(v, (int, float))]
            mean_metrics = {
                f"mean_{k}": float(np.mean([t[k] for t in batch_telemetry]))
                for k in numeric_keys
            }
            whole_energy = float(np.sum([t['energy_j'] for t in batch_telemetry]))
            mean_metrics.update({
                "ner_precision": ner_metrics["precision"],
                "ner_recall": ner_metrics["recall"],
                "ner_f1": ner_metrics["f1"],
                "whole_energy": whole_energy
            })
            mlflow.log_metrics(mean_metrics)

            mean_metrics_path = os.path.join(full_artifacts_dir, f"mean_metrics_B_{args.batch_size}.json")
            with open(mean_metrics_path, "w") as f:
                json.dump(mean_metrics, f, indent=2, default=numpy_serializer)
            mlflow.log_artifact(mean_metrics_path)

        print(f"Completed: F1={ner_metrics['f1']:.4f}, Latency={mean_metrics['avg_latency_s']:.4f}s, Energy={mean_metrics['avg_energy_j']:.4f}J, Whole Energy={mean_metrics['whole_energy']:.4f}J")

def main():
    parser = argparse.ArgumentParser(description="XTREME NER evaluation with vLLM + energy & MLflow")
    parser.add_argument(
        "--language",
        type=str,
        required=True,
        choices=["en", "de", "ar", "bg", "zh"],
        help="Language subset from XTREME PAN-X."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model identifier passed in the request payload (e.g., '/model' or 'my-deployed-model')."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Fixed batch size to use (default: 128)."
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=150,
        help="Max tokens to generate per request (default: 150)."
    )
    args = parser.parse_args()
    asyncio.run(run(args))

if __name__ == "__main__":
    main()
