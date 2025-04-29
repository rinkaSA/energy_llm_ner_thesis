import ray
import time
import json
import ast
import re
import mlflow
import numpy as np
from datasets import load_dataset
import evaluate
from openai import OpenAI
from dotenv import load_dotenv
import psutil
import GPUtil
import subprocess

load_dotenv()
try:
    with open("/data/horse/ws/irve354e-uniNer_test/energy_ner_llm/energy_ner_llm/mod_distil_by_cot/my_key_scads") as keyfile:
       my_api_key = keyfile.readline().strip()
except FileNotFoundError:
    print("file 'my_key_scads' was not found")
    exit(1)


def get_system_metrics():
    """
    Collect system metrics including CPU percentage, memory usage,
    GPU metrics via GPUtil, and power draw via nvidia-smi.
    """
    try:
        cpu_percent = psutil.cpu_percent(interval=0.5)
        mem = psutil.virtual_memory()
        memory_usage = {
            "total": mem.total,
            "available": mem.available,
            "percent": mem.percent,
            "used": mem.used,
            "free": mem.free
        }
    except Exception as e:
        cpu_percent = None
        memory_usage = {}
    
    try:
        gpus = GPUtil.getGPUs()
        gpu_metrics = []
        for gpu in gpus:
            gpu_metrics.append({
                "id": gpu.id,
                "load": gpu.load,
                "memoryUtil": gpu.memoryUtil,
                "memoryTotal": gpu.memoryTotal,
                "memoryUsed": gpu.memoryUsed,
                "powerDraw": gpu.powerDraw,
                "powerLimit": gpu.powerLimit,
                "temperature": gpu.temperature
            })
    except Exception as e:
        gpu_metrics = []
    
    try:
        nvidia_info = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader"],
            universal_newlines=True
        ).strip()
    except Exception as e:
        nvidia_info = "N/A"
    
    return {
        "cpu_percent": cpu_percent,
        "memory_usage": memory_usage,
        "gpu_metrics": gpu_metrics,
        "nvidia_smi_power": nvidia_info
    }

def np_encoder(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

def extract_json_with_split(generated_text, marker="Step 4:####"):
    """
    Attempt to extract the JSON (or Python dict literal) using a simple split on a marker.
    """
    if marker in generated_text:
        json_part = generated_text.split(marker)[-1].strip()
        try:
            result = json.loads(json_part)
            if isinstance(result, dict):
                return result
        except Exception:
            try:
                result = ast.literal_eval(json_part)
                if isinstance(result, dict):
                    return result
            except Exception:
                pass
    return None

def extract_last_json_block(generated_text):
    """
    Fallback: Use regex to extract all candidate JSON-like blocks and return the last one
    that can be parsed into a dictionary.
    """
    matches = re.findall(r"(\{.*?\})", generated_text, re.DOTALL)
    if matches:
        for candidate in reversed(matches):
            candidate = candidate.strip()
            try:
                result = json.loads(candidate)
                if isinstance(result, dict):
                    return result
            except Exception:
                try:
                    result = ast.literal_eval(candidate)
                    if isinstance(result, dict):
                        return result
                except Exception:
                    continue
    return {}

def parse_output(sample, output):
    """

    The generated text is expected to include multi-step reasoning where the final step contains
    a JSON or Python dictionary mapping entity types to lists of entity phrases.
    
    First, try to extract using a simple split with marker "Step 4:####".
    If that fails, fall back to a regex-based extraction.
    Returns:
      - labels: token-level BIO labels computed from the extracted dictionary.
      - reasoning: the full generated text (for inspection).
    """
    generated_text = output[0].get("generated_text", "")
    reasoning = generated_text.strip()
    
    predicted_entities_by_type = extract_json_with_split(generated_text, marker="Step 4:####")
    if predicted_entities_by_type is None:
        predicted_entities_by_type = extract_last_json_block(generated_text)
    
    tokens = sample["tokens"]
    labels = ["O"] * len(tokens)
    
    for entity_type, entities in predicted_entities_by_type.items():
        for entity in entities:
            if not isinstance(entity, str):
                continue
            entity_tokens = entity.split()
            n = len(entity_tokens)
            for i in range(len(tokens) - n + 1):
                if tokens[i:i+n] == entity_tokens:
                    labels[i] = f"B-{entity_type}"
                    for j in range(1, n):
                        labels[i+j] = f"I-{entity_type}"
    return labels, reasoning

def evaluate_and_log(all_true_labels, all_predicted_labels, comparisons, artifact_prefix):
    """
    Compute seqeval metrics and log overall as well as per-entity type scores.
    Save detailed results (including comparisons) as a JSON artifact.
    """
    seqeval_metric = evaluate.load("seqeval")
    results = seqeval_metric.compute(predictions=all_predicted_labels, references=all_true_labels)
    
    for key, value in results.items():
        if key.startswith("overall_"):
            mlflow.log_metric(key, value)
    
    for entity, scores in results.items():
        if isinstance(scores, dict):
            for metric, score in scores.items():
                mlflow.log_metric(f"{entity}_{metric}", score)
    
    detailed_results = {
        "comparisons": comparisons,
        "evaluation": results
    }
    detailed_json = f"{artifact_prefix}_detailed_results.json"
    with open(detailed_json, "w") as f:
        json.dump(detailed_results, f, indent=2, default=np_encoder)
    mlflow.log_artifact(detailed_json)
    
    return results


my_api_key = "your_api_key_here"
prompt_CoT_CONLL = "Your prompt template with a placeholder: {}"
label_names = {
    0: "O", 1: "B-PER", 2: "I-PER",
    3: "B-ORG", 4: "I-ORG",
    5: "B-LOC", 6: "I-LOC",
    7: "B-MISC", 8: "I-MISC"
}

client = OpenAI(base_url="https://llm.scads.ai/v1",api_key=my_api_key)
model_name = 'meta-llama/Llama-3.3-70B-Instruct'

def annotate_sample(joined_text):
    prompt = prompt_CoT_CONLL.format(joined_text)
    messages = [{'role': 'user', 'content': prompt}]
    response = client.chat.completions.create(model=model_name, messages=messages)
    generated_text = response.choices[0].message.content.strip()
    return [{"generated_text": generated_text}]

print(ray._private.services.get_node_ip_address())
# Initialize Ray on cluster (this connects to the head node)
ray.init(address="auto")

@ray.remote(num_gpus=1)
def annotate_sample_ray(sample):
    """
    Ray remote function to annotate a single sample.
    It times only the LLM call, then parses the output.
    """
    sys_metrics_before = get_system_metrics()
    joined_text = " ".join(sample["tokens"])
    start_llm = time.time()
    output = annotate_sample(joined_text)
    llm_time = time.time() - start_llm
    sys_metrics_after = get_system_metrics()
    predicted_labels, reasoning = parse_output(sample, output)
    result = {
        "text": joined_text,
        "true_labels": [label_names[tag] for tag in sample["ner_tags"]],
        "predicted_labels": predicted_labels,
        "raw_output": output,   
        "reasoning": reasoning, 
        "llm_time": llm_time,
        "system_metrics": {
            "before": sys_metrics_before,
            "after": sys_metrics_after
        }
    }
    return result



dataset = load_dataset("conll2003")["train"]
selected_dataset = dataset.shuffle(seed=12).select(range(1000))


# Launch annotation tasks in parallel using Ray.
futures = [annotate_sample_ray.remote(sample) for sample in selected_dataset]
results = ray.get(futures)


all_true_labels = []
all_predicted_labels = []
comparisons = []
total_llm_time = 0.0

for res in results:
    all_true_labels.append(res["true_labels"])
    all_predicted_labels.append(res["predicted_labels"])
    comparisons.append(res)
    total_llm_time += res["llm_time"]

mlflow.set_experiment("ConLL2003_Annotation_Experiment_distributed")
with mlflow.start_run(run_name='CoT_distributed', log_system_metrics=True) as run:
    mlflow.log_param("model", model_name)
    mlflow.log_param("dataset", "conll2003-train (1000 random samples)")
    mlflow.log_param("seed", 12)
    
    mlflow.log_metric("total_llm_time_seconds", total_llm_time)
    mlflow.log_metric("average_llm_time_per_sample_seconds", total_llm_time / len(selected_dataset))
    mlflow.log_metric("total_samples", len(selected_dataset))
    
    comparisons_file = "comparisons.json"
    with open(comparisons_file, "w") as f:
        json.dump(comparisons, f, indent=2, default=np_encoder)
    mlflow.log_artifact(comparisons_file)
    
    artifact_prefix = "conll_train_1000"
    eval_results = evaluate_and_log(all_true_labels, all_predicted_labels, comparisons, artifact_prefix)
    
    print(f"MLflow run completed. Run ID: {run.info.run_id}")
    print(f"Total LLM call time: {total_llm_time:.2f} seconds")
    print("Evaluation results:", eval_results)
