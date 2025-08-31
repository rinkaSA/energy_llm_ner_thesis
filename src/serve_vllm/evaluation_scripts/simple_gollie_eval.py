#!/usr/bin/env python3
import json
import os
import re
import asyncio
import aiohttp
from datasets import load_dataset
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
COMPLETIONS_URL = os.getenv("COMPLETIONS_URL")

if not COMPLETIONS_URL:
    raise RuntimeError("COMPLETIONS_URL environment variable not set")

# GoLLIE prompt header for English
ENGLISH_HEADER = '''from typing import List
from dataclasses import dataclass
import json

@dataclass
class PER:
    mention: str

@dataclass  
class ORG:
    mention: str

@dataclass
class LOC:
    mention: str

def annotate(text: str) -> List:
    """
    Given a text, annotate it with named entities.
    
    Args:
        text (str): The text to be annotated.
        
    Returns:
        List: A list of named entities. Each entity is an instance of PER, ORG, or LOC.
    """'''

def build_gollie_prompt(sentence: str) -> str:
    """Build GoLLIE prompt for a sentence."""
    text_literal = json.dumps(sentence, ensure_ascii=False)
    return ENGLISH_HEADER + f"\ntext = {text_literal}\n# Annotate entities in the given language.\nresult = [\n"

# Parse GoLLIE output
CLASS_PATTERNS = {
    "PER": re.compile(r'PER\s*\(\s*mention\s*=\s*(?P<q>["\'])(?P<val>.+?)\1\s*\)'),
    "ORG": re.compile(r'ORG\s*\(\s*mention\s*=\s*(?P<q>["\'])(?P<val>.+?)\1\s*\)'),
    "LOC": re.compile(r'LOC\s*\(\s*mention\s*=\s*(?P<q>["\'])(?P<val>.+?)\1\s*\)'),
}

def parse_gollie_output_to_entities(text: str) -> dict:
    """Parse GoLLIE output to extract entities."""
    if "]" in text:
        text = text.split("]", 1)[0]
    
    out = {"PER": [], "ORG": [], "LOC": []}
    for k, pat in CLASS_PATTERNS.items():
        vals = [m.group("val").strip() for m in pat.finditer(text)]
        # Remove duplicates while preserving order
        seen = set()
        dedup = []
        for v in vals:
            if v not in seen:
                seen.add(v)
                dedup.append(v)
        out[k] = dedup
    return out

def get_bio_tags(sentence, entities):
    """Convert sentence and entities to BIO tags."""
    tokens = sentence.split()
    tags = ["O"] * len(tokens)
    
    for ent_type, mentions in entities.items():
        for mention in mentions:
            mention_tokens = mention.split()
            for i in range(len(tokens) - len(mention_tokens) + 1):
                if tokens[i:i+len(mention_tokens)] == mention_tokens:
                    # Check if this span is already tagged to avoid conflicts
                    if all(tags[i+j] == "O" for j in range(len(mention_tokens))):
                        tags[i] = "B-" + ent_type
                        for j in range(1, len(mention_tokens)):
                            tags[i+j] = "I-" + ent_type
                    break
    
    return tokens, tags

async def send_completion_request(session, sentence, model_name="/gollie", max_tokens=150):
    """Send a single completion request."""
    prompt = build_gollie_prompt(sentence)
    payload = {
        "model": model_name,
        "prompt": prompt,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "stop": ["]\n", "\n]", "]"]
    }
    
    try:
        async with session.post(COMPLETIONS_URL, json=payload, timeout=aiohttp.ClientTimeout(total=60)) as resp:
            if resp.status != 200:
                return {"error": f"HTTP {resp.status}", "text": await resp.text()}
            return await resp.json()
    except Exception as e:
        return {"error": str(e)}

async def process_batch(session, sentences, model_name="/gollie", max_tokens=150):
    """Process a batch of sentences."""
    tasks = [send_completion_request(session, sentence, model_name, max_tokens) for sentence in sentences]
    return await asyncio.gather(*tasks, return_exceptions=True)

async def evaluate_xtreme_english(batch_size=128, model_name="/gollie", max_tokens=150):
    """Evaluate on XTREME English dataset."""
    print("Loading XTREME English dataset...")
    test_ds = load_dataset("google/xtreme", "PAN-X.en", split="test", trust_remote_code=True)
    test_subset = test_ds.select(range(min(batch_size, len(test_ds))))
    labels = test_ds.features["ner_tags"].feature.names
    
    print(f"Loaded {len(test_subset)} samples with labels: {labels}")
    
    all_gold = []
    all_pred = []
    results = []
    
    # Prepare batch
    sentences = []
    gold_tags_batch = []
    
    for example in test_subset:
        sentence = " ".join(example["tokens"])
        gold_tags = [labels[tag] for tag in example["ner_tags"]]
        sentences.append(sentence)
        gold_tags_batch.append(gold_tags)
    
    print(f"Processing batch of {len(sentences)} sentences...")
    
    async with aiohttp.ClientSession() as session:
        responses = await process_batch(session, sentences, model_name, max_tokens)
        
        for i, (response, gold_tags, sentence) in enumerate(zip(responses, gold_tags_batch, sentences)):
            # Extract text from response
            if isinstance(response, Exception):
                text = ""
                print(f"Error in response {i}: {response}")
            elif isinstance(response, dict) and "choices" in response and len(response["choices"]) > 0:
                text = response["choices"][0].get("text", "")
            else:
                text = str(response)
            
            # Parse entities
            entities = parse_gollie_output_to_entities(text)
            
            # Convert to BIO tags
            _, pred_tags = get_bio_tags(sentence, entities)
            
            # Align predictions with gold tags
            if len(pred_tags) < len(gold_tags):
                pred_tags += ["O"] * (len(gold_tags) - len(pred_tags))
            elif len(pred_tags) > len(gold_tags):
                pred_tags = pred_tags[:len(gold_tags)]
            
            all_gold.append(gold_tags)
            all_pred.append(pred_tags)
            
            results.append({
                "sentence": sentence,
                "raw_output": text,
                "entities": entities,
                "gold_tags": gold_tags,
                "pred_tags": pred_tags
            })
            
            if i < 5:  # Print first 5 examples for debugging
                print(f"\nExample {i+1}:")
                print(f"Sentence: {sentence[:100]}...")
                print(f"Raw output: {text[:100]}...")
                print(f"Entities: {entities}")
                print(f"Gold tags: {gold_tags[:10]}...")
                print(f"Pred tags: {pred_tags[:10]}...")
    
    # Compute metrics
    if len(all_gold) > 0 and len(all_pred) > 0:
        precision = precision_score(all_gold, all_pred)
        recall = recall_score(all_gold, all_pred)
        f1 = f1_score(all_gold, all_pred)
        report = classification_report(all_gold, all_pred, output_dict=True)
        
        metrics = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "classification_report": report
        }
    else:
        metrics = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "classification_report": {}}
    
    return metrics, results

async def main():
    """Main function."""
    print("Starting simple GoLLIE evaluation on XTREME English...")
    
    batch_size = 128
    model_name = "/gollie"
    max_tokens = 150
    
    metrics, results = await evaluate_xtreme_english(
        batch_size=batch_size,
        model_name=model_name,
        max_tokens=max_tokens
    )
    
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Processed {len(results)} examples")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1']:.4f}")
    
    print("\nPer-entity results:")
    for entity_type in ["PER", "ORG", "LOC"]:
        if entity_type in metrics["classification_report"]:
            entity_metrics = metrics["classification_report"][entity_type]
            print(f"{entity_type}: P={entity_metrics['precision']:.4f}, "
                  f"R={entity_metrics['recall']:.4f}, "
                  f"F1={entity_metrics['f1-score']:.4f}")
    
    # Save results
    output_file = "simple_gollie_eval_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "metrics": metrics,
            "results": results,
            "config": {
                "batch_size": batch_size,
                "model_name": model_name,
                "max_tokens": max_tokens
            }
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    asyncio.run(main())
