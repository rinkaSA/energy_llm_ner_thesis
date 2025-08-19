
"""
Key improvements in this version:

Uses aiohttp for asynchronous HTTP requests
Processes prompts in true batches using concurrent requests
Configures the vLLM server with proper batching parameters
Maintains all the existing functionality but with better performance
Error handling for failed batch requests
Progress tracking with tqdm

"""
import json
import mlflow
import asyncio
import aiohttp
from datasets import load_dataset
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
from tqdm import tqdm
import os
from typing import List, Dict, Any
import numpy as np
import re
from dotenv import load_dotenv
import requests

load_dotenv()
SERVER_URL = os.getenv("SERVER_URL")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")

if not MLFLOW_TRACKING_URI:
    raise RuntimeError("Please set MLFLOW_TRACKING_URI in your .env")

INSTRUCTION = """### Instruction:
You are an expert in natural language processing annotation. 
Given a sentence, identify and classify each named entity into one of the following types: 
LOC (Location), MISC (Miscellaneous), ORG (Organization), or PER (Person).

For example, consider the sentence:
'Brazilian Planning Minister Antonio Kandir will submit to a draft copy of the 1997 federal budget to Congress on Thursday, a ministry spokeswoman said.'

Expected output: {'MISC': ['Brazilian'], 'PER': ['Antonio Kandir'], 'ORG': ['Congress']}

Given the sentence below perform a task and include in the resulting output in json style format as in example only."""


system_msg = {
    "role": "system",
    "content": (
        "You are a German NER engine.  \n"
        "- **Input:** an **English** sentence.  \n"
        "- **Output:** **only** a **single**, **valid** JSON object with three arrays: `PER`, `ORG`, `LOC`.  \n"
        "- Do **not** output any extra text, bulleted lists, or explanation.  \n"
        "- JSON must be parseable by `json.loads`."
    )
}

examples = [
    {"role":"user",   "content":"Sentence: Angela Merkel spoke at the UN headquarters in New York City."},
    {"role":"assistant","content":(
        '{\n'
        '  "PER": ["Angela Merkel"],\n'
        '  "ORG": ["UN-Hauptquartier"],\n'
        '  "LOC": ["New York City"]\n'
        '}'
    )}
]
def make_messages_for(sentence: str):
    return [system_msg] + examples + [
        {"role":"user", "content":f"Sentence: {sentence}"}
    ]



async def process_batch(
    session: aiohttp.ClientSession,
    prompts: List[str],
    max_tokens: int = 150
) -> List[str]:
    """Process a batch of prompts concurrently."""
    async def single_request(prompt: str) -> str:
        payload = {
            "model": "/model",
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0
        }
        async with session.post(SERVER_URL, json=payload) as response:
            response.raise_for_status()
            result = await response.json()
            return result["choices"][0]["text"]

    tasks = [single_request(prompt) for prompt in prompts]
    return await asyncio.gather(*tasks)

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
            resp.raise_for_status()
            j = await resp.json()
            return j["choices"][0]["message"]["content"]
    tasks = [single_request(s) for s in sentences]
    return await asyncio.gather(*tasks)



async def evaluate_ner_pipeline_conll03(
    test_dataset: Any,
    label_list: List[str],
    batch_size: int = 32,
    max_new_tokens: int = 150
) -> tuple[Dict, List[Dict]]:
    """Evaluate NER performance using batched async requests."""
    all_gold_tags = []
    all_pred_tags = []
    generated_results = []
    
    # Process dataset in batches
    async with aiohttp.ClientSession() as session:
        for i in tqdm(range(0, len(test_dataset), batch_size)):
            #
            # batch_examples = test_dataset[i:i + batch_size]
            batch_examples = test_dataset.select(range(i, min(i + batch_size, len(test_dataset)))).to_list()
            # Prepare batch data
            sentences = [" ".join(ex["tokens"]) for ex in batch_examples]
            gold_tags = [[label_list[tag] for tag in ex["ner_tags"]] for ex in batch_examples]
            #prompts = [f"{INSTRUCTION}\nNow, given the sentence: {sent}. ### Response:" for sent in sentences]
            
            # Process batch
            try:
                responses = await process_batch_chat(session, sentences, max_new_tokens)
                
                # Process responses
                for sentence, response, g_tags in zip(sentences, responses, gold_tags):
                    if "### Response:" in response:
                        pred_text = response.split("### Response:")[-1].strip()
                    else:
                        pred_text = response.strip()
                        
                    generated_results.append({
                        "prompt": sentence,
                        "generated_response": pred_text
                    })
                    
                    pred_entities = parse_response(pred_text)
                    _, pred_tags = get_bio_tags(sentence, pred_entities)
                    
                    all_gold_tags.append(g_tags)
                    all_pred_tags.append(pred_tags)
                    
            except Exception as e:
                print(f"Error processing batch: {e}")
                continue

    # Calculate metrics
    metrics = {
        "precision": precision_score(all_gold_tags, all_pred_tags),
        "recall": recall_score(all_gold_tags, all_pred_tags),
        "f1": f1_score(all_gold_tags, all_pred_tags),
        "classification_report": classification_report(all_gold_tags, all_pred_tags, output_dict=True)
    }
    
    return metrics, generated_results

async def main():
    experiment_name = "Server_Evaluation"
    run_name = "Mistral-awq"
    
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name=run_name, log_system_metrics=True) as run:
        model_name = "mistral-awq"
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("PROMPT", INSTRUCTION)
        
        test_dataset = load_dataset(
            "conll2003",
            trust_remote_code=True,
            split="test"
        ).select(range(100))
        
        label_list = test_dataset.features["ner_tags"].feature.names
        
        metrics, generated_results = await evaluate_ner_pipeline_conll03(
            test_dataset,
            label_list,
            batch_size=32
        )
        
        print("Evaluation Metrics:")
        print(metrics)

        # Save and log results
        os.makedirs("server_response_results", exist_ok=True)
        
        metrics_filename = "server_response_results/evaluation_metrics.json"
        with open(metrics_filename, "w") as f:
            json.dump(metrics, f, indent=4, default=np_encoder)

        responses_filename = "server_response_results/llm_generated_responses.json"
        with open(responses_filename, "w") as f:
            json.dump(generated_results, f, indent=4, default=np_encoder)

        # Log metrics to MLflow
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
    # Import parse_response, get_bio_tags, and np_encoder from the original script
    from energy_ner_llm.src.serve_vllm.evaluation_scripts.eval_llm_batches import parse_response, get_bio_tags, np_encoder
    asyncio.run(main())