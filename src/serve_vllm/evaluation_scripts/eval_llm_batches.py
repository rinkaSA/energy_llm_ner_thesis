import json
import mlflow
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig  # These may be unused now.
from datasets import load_dataset, DownloadConfig
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
from tqdm import tqdm
import os
import requests  # For sending HTTP requests to the remote server
import numpy as np
import re
import ast
from dotenv import load_dotenv
from huggingface_hub import login

load_dotenv()
SERVER_URL = os.getenv("SERVER_URL")
#SERVER_URL = 'http://c106.capella.hpc.tu-dresden.de:8000/v1/completions'
if not SERVER_URL:
    raise RuntimeError("SERVER_URL environment variable not set")
#MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")

#if not MLFLOW_TRACKING_URI: 
 #   raise RuntimeError("Please set MLFLOW_TRACKING_URI in your .env")


def np_encoder(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")
    
def parse_response(response_text):
    """
    Parses the generated response to extract entity mentions in a consistent format.
    
    It now supports three formats:
      1. A pure dictionary string, e.g.:
         "{'LOC': ['JAPAN', 'CHINA']}"
      2. A dictionary string prefixed or suffixed with extra text, e.g.:
         "Expected output: {'MISC': ['China', 'Uzbekistan'], 'PER': ['newcomers'], 'ORG': []}\n\nPlease provide..."
      3. A line-based format, e.g.:
         "LOC - China\nMISC - newcomers\nORG - Uzbekistan\nPER - none"
         
    Returns a dictionary mapping entity types ("LOC", "MISC", "ORG", "PER") to lists of entity mentions.
    """
    stripped = response_text.strip()
    # Attempt to find the first dictionary by searching for the first '{'
    start_index = stripped.find("{")
    if start_index != -1:
        bracket_count = 0
        end_index = None
        for i in range(start_index, len(stripped)):
            if stripped[i] == '{':
                bracket_count += 1
            elif stripped[i] == '}':
                bracket_count -= 1
                if bracket_count == 0:
                    end_index = i
                    break
        if end_index is not None:
            dict_part = stripped[start_index:end_index+1]
            try:
                parsed = ast.literal_eval(dict_part)
                if isinstance(parsed, dict):
                    new_dict = {}
                    for k, v in parsed.items():
                        key = k.strip().upper()
                        if isinstance(v, str):
                            v = [x.strip() for x in v.split(",") if x.strip()]
                        elif isinstance(v, list):
                            v = [str(x).strip() for x in v]
                        new_dict[key] = v
                    return new_dict
            except Exception:
                pass  # fall back to line-based parsing if extraction fails

    # Line-based parsing
    entities = {}
    if "\n" in response_text:
        lines = response_text.strip().splitlines()
    elif ";" in response_text:
        lines = [part.strip() for part in response_text.split(";") if part.strip()]
    else:
        lines = response_text.strip().splitlines()

    for line in lines:
        if " - " in line:
            parts = line.split(" - ", 1)
        elif ":" in line:
            parts = line.split(":", 1)
        else:
            continue  
        if len(parts) != 2:
            continue
        
        raw_entity_type, entity_value = parts
        entity_type = raw_entity_type.split(":")[0].split("-")[0].strip().upper()
        entity_value = entity_value.strip().rstrip(".")
        if entity_value.lower() == "none":
            continue

        mentions = [mention.strip() for mention in entity_value.split(",") if mention.strip()]
        if mentions:
            if entity_type in entities:
                entities[entity_type].extend(mentions)
            else:
                entities[entity_type] = mentions
    return entities

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

def send_inference_request(prompt, max_new_tokens=150):
    """
    Send an HTTP POST request to the remote inference server.
    
    Parameters:
      prompt (str): The prompt to send.
      max_new_tokens (int): The maximum number of new tokens to request.
    
    Returns:
      dict: The JSON response from the inference server.
    """

    payload = {
        "model": "/model",
        "prompt": prompt,
        "max_tokens": max_new_tokens,
        "temperature": 0.0  
    }
    response = requests.post(SERVER_URL, json=payload)
    response.raise_for_status()
    data = response.json()
    print(f"gen: {data['choices'][0]['text']}")
    return data["choices"][0]["text"]

def evaluate_ner_pipeline_conll03(test_dataset, label_list, max_new_tokens=100, batch_size=8):
    """
    Evaluate NER performance on the CoNLL03 test set using remote inference.
    
    Parameters:
      test_dataset: The CoNLL03 test dataset loaded from Hugging Face.
      label_list: A list mapping numeric ner_tags to string labels.
      max_new_tokens (int): Maximum tokens to generate.
      batch_size (int): Number of examples to process in each batch.
    
    Returns:
      A tuple of:
        - metrics (dict): Contains "precision", "recall", "f1", and "classification_report".
        - generated_results (list): List of dictionaries containing prompts and generated responses.
    """
    all_gold_tags = []
    all_pred_tags = []
    prompts_batch = []
    gold_sentences_batch = []
    generated_results = []
    
    instruction = (
    "### Instruction:\nYou are an expert in natural language processing annotation. "
    "Given a sentence, identify and classify each named entity into one of the following types: "
    "LOC (Location), MISC (Miscellaneous), ORG (Organization), or PER (Person).\n\n"
    "For example, consider the sentence:\n"
    "'Brazilian Planning Minister Antonio Kandir will submit to a draft copy of the 1997 federal budget "
    "to Congress on Thursday, a ministry spokeswoman said.'\n\n"
    "Expected output: {'MISC': ['Brazilian'], 'PER': ['Antonio Kandir'], 'ORG': ['Congress']}\n\n"
     " Given the sentence below perform a task and include in the resulting output in json style format as in example only.")

    mlflow.log_param("PROMPT", instruction)
    for example in tqdm(test_dataset, desc="Evaluating CoNLL03"):
        sentence = " ".join(example["tokens"])
        gold_tags = [label_list[tag] for tag in example["ner_tags"]]
        
        prompt = instruction + f"Now, given the sentence: {sentence}. ### Response:"
        
        prompts_batch.append(prompt)
        gold_sentences_batch.append((sentence, gold_tags))
        
        if len(prompts_batch) == batch_size:
            # Instead of using pipeline generator, we call the remote server for each prompt
            outputs = [send_inference_request(prompt, max_new_tokens) for prompt in prompts_batch]
            for out, (sentence, gold_tags) in zip(outputs, gold_sentences_batch):
                # Expecting the response to be a JSON with a key "generated_text"
                generated_text = out
                if "### Response:" in generated_text:
                    pred_text = generated_text.split("### Response:")[-1].strip()
                else:
                    pred_text = generated_text.strip()
                generated_results.append({
                    "prompt": sentence,
                    "generated_response": pred_text
                })
                pred_entities = parse_response(pred_text)
                _, pred_tags = get_bio_tags(sentence, pred_entities)
                all_gold_tags.append(gold_tags)
                all_pred_tags.append(pred_tags)
            prompts_batch = []
            gold_sentences_batch = []
    
    # Process any remaining prompts
    if prompts_batch:
        outputs = [send_inference_request(prompt, max_new_tokens) for prompt in prompts_batch]
        for out, (sentence, gold_tags) in zip(outputs, gold_sentences_batch):
            generated_text = out
            if "### Response:" in generated_text:
                pred_text = generated_text.split("### Response:")[-1].strip()
            else:
                pred_text = generated_text.strip()
            pred_entities = parse_response(pred_text)
            _, pred_tags = get_bio_tags(sentence, pred_entities)
            all_gold_tags.append(gold_tags)
            all_pred_tags.append(pred_tags)
    
    precision = precision_score(all_gold_tags, all_pred_tags)
    recall = recall_score(all_gold_tags, all_pred_tags)
    f1 = f1_score(all_gold_tags, all_pred_tags)
    report = classification_report(all_gold_tags, all_pred_tags, output_dict=True)
    
    metrics = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "classification_report": report
    }
    return metrics, generated_results

def main():
    

    experiment_name = "Server_Evaluation"
    run_name = "mistral_awq"
    
    #mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name=run_name, log_system_metrics=True) as run:
        model_name = "meta-llama/Llama-2-7b-chat-hf"
        mlflow.log_param("model_name", model_name)
        
        # Optionally: skip local model and tokenizer loading if they are not used.
        # tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        # tokenizer.pad_token = tokenizer.eos_token
        # base_model = AutoModelForCausalLM.from_pretrained(
        #     model_name,
        #     trust_remote_code=True
        # )
        # base_model.eval()

        # dl_config = DownloadConfig(verify_ssl=False) , download_config=dl_config

        test_dataset = load_dataset("conll2003",trust_remote_code=True,
                                    split="test").select(range(20))
        label_list = test_dataset.features["ner_tags"].feature.names

        metrics, generated_results = evaluate_ner_pipeline_conll03(test_dataset, label_list, batch_size=8)
        print("Evaluation Metrics:")
        print(metrics)

        metrics_filename = "server_response_results/evaluation_metrics.json"
        with open(metrics_filename, "w") as f:
            json.dump(metrics, f, indent=4, default=np_encoder)

        responses_filename = "server_response_results/llm_generated_responses.json"
        with open(responses_filename, "w") as f:
            json.dump(generated_results, f, indent=4, default=np_encoder)

        mlflow.log_metric("precision", metrics["precision"])
        mlflow.log_metric("recall", metrics["recall"])
        mlflow.log_metric("f1", metrics["f1"])
        for entity, scores in metrics["classification_report"].items():
            if isinstance(scores, dict):
                for score_name, value in scores.items():
                    metric_name = f"{entity}_{score_name}"
                    metric_name = re.sub(r'[^a-zA-Z0-9_\-\. /]', '', metric_name)
                    mlflow.log_metric(metric_name, value)
        
        mlflow.log_artifact(metrics_filename)
        mlflow.log_artifact(responses_filename)
        
if __name__ == "__main__":
    main()
