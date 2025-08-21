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
ENERGY_METRIC_NAME = "DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION"  
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
FEW_SHOTS_PATH = os.getenv("FEW_SHOTS_PATH", "ner_few_shots_masakha.json")

### IN THIS DATASET PER, LOC, ORG, AND DATE!! ARE GOLD TAGS. I get rid of date for now!

TARGET_TYPES = {"PER", "ORG", "LOC"}


masakhaner2_langs = {
    "bam": "Bambara",
    "ewe": "Ewe",
    "fon": "Fon",
    "hau": "Hausa",
    "ibo": "Igbo",
    "kin": "Kinyarwanda",
    "lug": "Luganda",
    "luo": "Luo (Dholuo)",
    "mos": "Mossi",
    "pcm": "Nigerian Pidgin",
    "sna": "Shona",
    "swa": "Swahili",
    "tsn": "Setswana (Tswana)",
    "twi": "Twi (Akan)",
    "wol": "Wolof",
    "xho": "Xhosa",
    "yor": "Yoruba",
    "zul": "Zulu"
}

SYSTEM_TEMPLATE = (
    "You are a Named Entity Recognition (NER) annotator for the {language} language.\n"
    "- Input: a single {language} sentence (do not translate).\n"
    "- Output: only a single, valid JSON object with three arrays: `PER`, `ORG`, `LOC`.\n"
    "- Do not output any extra text, explanations, or markdown.\n"
    "- JSON must be parseable by json.loads.\n"
    "- Extract only entities that appear verbatim in the input sentence (no hallucinations).\n"
    "- Keep entity surface forms exactly as they appear; do not normalize or translate."
)


def build_system_msg(language: str) -> dict:
    return {
        "role": "system",
        "content": SYSTEM_TEMPLATE.format(language=masakhaner2_langs.get(language, language))
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

async def process_batch_chat(session, sentences, max_tokens,model_name, language):
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
            return result["choices"][0]["message"]["content"]
    
    tasks = [single_request(s) for s in sentences]
    return await asyncio.gather(*tasks)

def project_to_targets(tags, target_types=TARGET_TYPES):
    """Keep only BIO tags whose entity type is in target_types; map others (e.g., DATE) to 'O'."""
    out = []
    for t in tags:
        if t == "O" or not t:
            out.append("O")
            continue
        parts = t.split("-", 1)
        if len(parts) != 2:
            out.append("O")
            continue
        pref, typ = parts[0], parts[1].upper()
        out.append(f"{pref}-{typ}" if typ in target_types else "O")
    return out


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


async def process_and_measure(session, prompts, max_tokens, model_name, language):
    e0 = read_energy_joules()
    v0 = read_vllm_metrics()
    t0 = time.perf_counter()

    responses = await process_batch_chat(session, prompts, max_tokens, model_name, language)

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


async def evaluate_ner_pipeline_masakhaner2(
    test_dataset, label_list, batch_size, model_name, max_new_tokens=150, language="twi"
):
    all_gold, all_pred = [], []
    generated_results = []
    batch_telemetry = []  

    async with aiohttp.ClientSession() as session:
        for i in tqdm(range(0, len(test_dataset), batch_size)):
            batch = test_dataset.select(range(i, min(i+batch_size, len(test_dataset)))).to_list()
            sents = [" ".join(ex["tokens"]) for ex in batch]
            
            golds_raw = [[label_list[t] for t in ex["ner_tags"]] for ex in batch]
            golds = [project_to_targets(seq) for seq in golds_raw]
            try:
                responses, telem = await process_and_measure(session, sents, max_new_tokens, model_name, language)
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


def build_artifacts_dir(language: str, batch_size: int, model_name: str):
    safe_model = re.sub(r"[^a-zA-Z0-9_.-]+", "_", model_name)
    return f"{language}_B{batch_size}_{safe_model}"



async def run(args):
    test_ds = load_dataset("masakhane/masakhaner2", args.language, split="test")
    labels = test_ds.features["ner_tags"].feature.names

    print(f"Loaded MasakhaNER2 {args.language} dataset with {len(test_ds)} samples and {len(labels)} labels: {labels}")

    test_subset = test_ds.select(range(min(1000, len(test_ds))))
    print(f"Testing with {len(test_subset)} samples")

    artifacts_dir = build_artifacts_dir(args.language, args.batch_size, args.model)
    base_dir = "./inference_eval_artifacts/masakha"
    full_artifacts_dir = os.path.join(base_dir, artifacts_dir)
    os.makedirs(full_artifacts_dir, exist_ok=True)


    experiment_name = f"{args.model}_masakha"
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.start_run(experiment_name=experiment_name)

    with mlflow.start_run(run_name=f"masakha_ner_{args.language}_B{args.batch_size}"):
        mlflow.set_tags({
            "language": args.language,
            "batch_size": str(args.batch_size),
            "model": args.model,
            "dataset": "masakha",
            "n_samples" : str(len(test_subset))
        })
        mlflow.log_params({
            "language": args.language,
            "batch_size": args.batch_size,
            "model": args.model,
            "max_new_tokens": args.max_new_tokens,
            "dataset": "masakha",
            "n_samples" : str(len(test_subset))
        })


        ner_metrics, gen_responses, batch_telemetry = await evaluate_ner_pipeline_masakhaner2(
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

        mlflow.log_artifact(responses_path)
        mlflow.log_artifact(ner_metrics_path)
        mlflow.log_artifact(telemetry_path)

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
    parser = argparse.ArgumentParser(description="MasakhaNER2 evaluation with vLLM + energy & MLflow")
    parser.add_argument(
        "--language",
        type=str,
        required=True,
        choices=["bam", "ewe", "fon", "hau", "ibo", "kin", "lug", "luo", "mos", "pcm", "sna", "swa", "tsn", "twi", "wol", "xho", "yor", "zul"],
        help="Language subset from MasakhaNER2 with bbj (Ghomala) and nya (Nyanja) exluded"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model identifier passed in the request payload ( '/model' or '/Mistral-7B-Instruct-v0.2')."
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

