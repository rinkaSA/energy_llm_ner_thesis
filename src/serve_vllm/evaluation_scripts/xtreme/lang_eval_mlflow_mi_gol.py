#!/usr/bin/env python3
import json
import os
import re
import ast
import time, datetime
import asyncio
import aiohttp
import requests
import numpy as np
import sys
from tqdm import tqdm
from datasets import load_dataset
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
from dotenv import load_dotenv
import argparse
import mlflow
import pandas as pd
from math import isnan
import torch




script_dir = os.path.dirname(os.path.abspath(__file__))
serve_vllm_dir = os.path.join(script_dir, '../../')
sys.path.insert(0, os.path.abspath(serve_vllm_dir))

# ---------------------------
# Env & constants
# ---------------------------
load_dotenv()

CHAT_URL = os.getenv("CHAT_URL")
COMPLETIONS_URL = os.getenv("COMPLETIONS_URL")
ENERGY_URL = os.getenv("ENERGY_URL")
VLLM_METRICS_URL = os.getenv("VLLM_METRICS_URL")
ENERGY_METRIC_NAME = "DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
FEW_SHOTS_PATH = os.path.join(serve_vllm_dir, "prompts_in_all_languages/ner_few_shots_full_panx.json")
metrics_csv_path = os.environ.get("METRICS_CSV")
job_id = os.getenv("JOB_ID")

print(CHAT_URL)
print(job_id)

# ---------- GoLLIE prompt headers ----------
# Ensure this file exists with: ENGLISH_HEADER, GERMAN_HEADER, CHINESE_HEADER, ARABIC_HEADER, BULGARIAN_HEADER
from prompts_in_all_languages.gollie_prompts import (
    ENGLISH_HEADER, GERMAN_HEADER, CHINESE_HEADER, ARABIC_HEADER, BULGARIAN_HEADER
)
LANG2HEADER = {
    "de": GERMAN_HEADER,
    "en": ENGLISH_HEADER,
    "zh": CHINESE_HEADER,
    "ar": ARABIC_HEADER,
    "bg": BULGARIAN_HEADER,
    # For other languages without specific headers, we'll handle them at runtime
    # by replacing placeholders in the template headers
}

# ---------- dataset helpers ----------
def resolve_xtreme_subset(language: str) -> str:
    """
    Maps language code to XTREME dataset subset name.
    
    For PAN-X datasets, the format is "PAN-X.<lang_code>"
    
    Args:
        language: ISO 639-1 language code (e.g., 'en', 'de')
    
    Returns:
        The XTREME dataset subset name
    
    Raises:
        ValueError: If language is not supported in XTREME
    """
    # Currently, only PAN-X dataset subsets are used for NER evaluation
    # XTREME supports more languages, but this function only handles the ones with NER data
    mapping = {
        "ar": "PAN-X.ar",
        "bg": "PAN-X.bg",
        "de": "PAN-X.de", 
        "en": "PAN-X.en",
        "es": "PAN-X.es", 
        "fr": "PAN-X.fr",
        "el": "PAN-X.el",
        "hi": "PAN-X.hi",
        "id": "PAN-X.id",
        "it": "PAN-X.it",
        "ja": "PAN-X.ja",
        "ko": "PAN-X.ko",
        "nl": "PAN-X.nl",
        "pt": "PAN-X.pt",
        "ru": "PAN-X.ru",
        "th": "PAN-X.th",
        "tr": "PAN-X.tr",
        "ur": "PAN-X.ur",
        "vi": "PAN-X.vi",
        "zh": "PAN-X.zh",
        "yo": "PAN-X.yo"
    }
    if language not in mapping:
        raise ValueError(f"Unsupported language: {language}. Available languages: {', '.join(sorted(mapping.keys()))}")
    return mapping[language]

def build_artifacts_dir(language: str, batch_size: int, model_name: str):
    safe_model = re.sub(r"[^a-zA-Z0-9_.-]+", "_", model_name)
    return f"xtreme_{language}_B{batch_size}_{safe_model}_{job_id}"

# ---------- Mistral (chat) prompt + parser ----------

SYSTEM_TEMPLATE = (
    "You are a Named Entity Recognition (NER) annotator for the {language} language.\n"
    "- Input: a single {language} sentence (do not translate).\n"
    "- Output: only a single, valid JSON object with only three arrays: `PER`, `ORG`, `LOC`.\n"
    "- Do not output any extra text, explanations, or markdown."
    "- Find entities of mentioned categories only (PER- person, ORG- organization, LOC- location).\n"
    "- JSON must be parseable by json.loads.\n"
    "- Extract only entities that appear verbatim in the input sentence (no hallucinations).\n"
    "- Keep entity surface forms exactly as they appear; do not normalize or translate."
)


def full_lang_name(language: str) -> str:
    """
    Returns the full language name for a given language code.
    
    This function maps ISO 639-1 language codes to their full language names.
    It supports all 40 languages in the XTREME benchmark.
    
    Args:
        language: ISO 639-1 language code (e.g., 'en', 'de')
    
    Returns:
        Full language name (e.g., 'English', 'German')
    """
    return {
        # Indo-European languages
        # Germanic
        "af": "Afrikaans",
        "de": "German", 
        "en": "English",
        "nl": "Dutch",
        # Romance
        "es": "Spanish",
        "fr": "French", 
        "it": "Italian",
        "pt": "Portuguese",
        # Slavic
        "bg": "Bulgarian",
        "ru": "Russian",
        # Indo-Aryan
        "bn": "Bengali",
        "hi": "Hindi",
        "mr": "Marathi",
        "ur": "Urdu",
        # Iranian
        "fa": "Persian",
        # Other Indo-European
        "el": "Greek",
        
        # Sino-Tibetan
        "zh": "Chinese",
        "my": "Burmese",
        
        # Afro-Asiatic
        "ar": "Arabic",
        "he": "Hebrew",
        
        # Japonic
        "ja": "Japanese",
        
        # Koreanic (isolate)
        "ko": "Korean",
        
        # Turkic
        "kk": "Kazakh",
        "tr": "Turkish",
        
        # Uralic
        "et": "Estonian",
        "fi": "Finnish",
        "hu": "Hungarian",
        
        # Austronesian
        "id": "Indonesian",
        "jv": "Javanese", 
        "ms": "Malay",
        "tl": "Tagalog",
        
        # Dravidian
        "ml": "Malayalam",
        "ta": "Tamil",
        "te": "Telugu",
        
        # Niger-Congo
        "sw": "Swahili",
        "yo": "Yoruba",
        
        # Kartvelian
        "ka": "Georgian",
        
        # Kra-Dai
        "th": "Thai",
        
        # Austro-Asiatic
        "vi": "Vietnamese",
        
        # Language isolate
        "eu": "Basque"
    }.get(language, language)

# ---------- Gemma prompt + parser ----------
def load_gemma_system_template(language:str, system_prompt_choice:int, use_generic_template=False) -> str:
    """Load a system prompt template for Gemma from JSON file by choice number.
    
    Args:
        language: Language code (e.g., 'en', 'de')
        system_prompt_choice: Which prompt variation to use (1-8)
        use_generic_template: If True, use Template_system_prompt.json instead of language-specific file
    """
    if use_generic_template:
        template_path = os.path.join(serve_vllm_dir, "prompts_in_all_languages/Template_system_prompt.json")
    else:
        template_path = os.path.join(serve_vllm_dir, f"prompts_in_all_languages/{full_lang_name(language)}_system_prompt.json")
    
    with open(template_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if str(system_prompt_choice) not in data:
        raise ValueError(f"No system prompt found for choice: {system_prompt_choice}")
    else:
        prompt = data[str(system_prompt_choice)] if isinstance(data[str(system_prompt_choice)], str) else "\n".join(data[str(system_prompt_choice)])

    # If using template, replace {language} placeholder with actual language name
    if use_generic_template:
        prompt = prompt.replace("{language}", full_lang_name(language))

    return prompt


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

def parse_response_json_like(response_text: str) -> dict:
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
            if isinstance(parsed, dict):
                break
        except Exception:
            continue

    result = {}
    if isinstance(parsed, dict):
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


# ---------- GoLLIE (code) prompt + parser ----------
def build_gollie_prompt(sentence: str, language: str, use_generic_template=False) -> str:
    """
    Build a prompt for GoLLIE model
    
    Args:
        sentence: Input sentence to analyze
        language: ISO 639-1 language code (e.g., 'en', 'de')
        use_generic_template: Whether to use a generic template with language injection
    
    Returns:
        The complete prompt for the GoLLIE model
    """
    # Use language-specific header if available
    if language in LANG2HEADER and not use_generic_template:
        header = LANG2HEADER[language]
    else:
        # For languages without specific headers, use English header with language name injected
        template_header = ENGLISH_HEADER
        header = template_header.replace("English language", f"{full_lang_name(language)} language")
    
    text_literal = json.dumps(sentence, ensure_ascii=False)
    # model completes after 'result = ['
    return header + f"\ntext = {text_literal}\n# Annotate entities in the {full_lang_name(language)} language.\nresult = [\n"

# ---------- Gemma prompt + parser ----------
def save_messages_to_file(messages_list, out_dir, model_name, language, batch_size):
    """
    Save messages to a JSON file for later analysis and logging
    """
    messages_dir = os.path.join(out_dir, "messages")
    os.makedirs(messages_dir, exist_ok=True)
    
    messages_path = os.path.join(messages_dir, f"{model_name.replace('/', '')}_messages_{language}_B{batch_size}.json")
    
    with open(messages_path, "w", encoding="utf-8") as f:
        json.dump(messages_list, f, indent=2, ensure_ascii=False)
    
    return messages_path

def build_gemma_messages(sentence: str, language: str, system_prompt_choice: int, examples_for_lang: list, use_generic_template=False) -> list:
    """
    Builds a list of messages for Gemma models in the chat format.
    Returns a list of messages with roles and content.
    
    Args:
        sentence: The input sentence to analyze
        language: Language code (e.g., 'en', 'de')
        system_prompt_choice: Which prompt variation to use (1-8)
        examples_for_lang: Few-shot examples for the language
        use_generic_template: If True, use Template_system_prompt.json instead of language-specific file
    """
    # Get the system prompt based on choice
    sys_msg = load_gemma_system_template(
        language=language, 
        system_prompt_choice=system_prompt_choice, 
        use_generic_template=use_generic_template
    )
    
    # Create system message
    messages = [{"role": "system", "content": sys_msg}]
    
    # Set number of few-shot examples based on system prompt choice
    pairs_for_choice = {
        1: 0,  # no shots
        2: 1,  # 1 pair
        3: 0,  # constraints, no shots
        4: 1,  # constraints + 1 pair
        5: 2,  # constraints + 2 pairs
        6: 3,  # constraints + 2 pairs + 1 hard-neg 
        7: 0,  # rules only
        8: 0,  # CoT demo inline in system, kinda few shot already
    }
    n_pairs = pairs_for_choice.get(system_prompt_choice, 0)
    
    # Add few-shot examples if needed
    if n_pairs > 0:
        # Take only the requested number of example pairs
        for i in range(min(n_pairs * 2, len(examples_for_lang))):
            messages.append(examples_for_lang[i])
    
    # Add the user query with the sentence
    messages.append({"role": "user", "content": sentence})
    
    return messages

CLASS_PATTERNS = {
    "PER": re.compile(r'PER\s*\(\s*mention\s*=\s*(?P<q>["\'])(?P<val>.+?)\1\s*\)'),
    "ORG": re.compile(r'ORG\s*\(\s*mention\s*=\s*(?P<q>["\'])(?P<val>.+?)\1\s*\)'),
    "LOC": re.compile(r'LOC\s*\(\s*mention\s*=\s*(?P<q>["\'])(?P<val>.+?)\1\s*\)'),
}
def parse_gollie_output_to_entities(text: str) -> dict:
    if "]" in text: text = text.split("]", 1)[0]
    out = {"PER": [], "ORG": [], "LOC": []}
    for k, pat in CLASS_PATTERNS.items():
        vals = [m.group("val").strip() for m in pat.finditer(text)]
        # dedup preserve order
        seen=set(); dedup=[]
        for v in vals:
            if v not in seen: seen.add(v); dedup.append(v)
        out[k]=dedup
    return out

def robust_gemma_entity_extraction(text: str) -> dict:
    """
    Extracts PER, ORG, LOC lists from any text, even if JSON is incomplete or broken.
    Handles malformed model outputs with missing brackets or incomplete JSON.
    Also handles code block markers.
    """
    # Clean up code block markers if present
    if "```json" in text:
        # Extract the JSON content between the markers
        json_match = re.search(r'```json\s*\n(.*?)```', text, re.DOTALL)
        if json_match:
            text = json_match.group(1).strip()
        else:
            # If no closing marker, take everything after the opening marker
            text = re.sub(r'```json\s*\n', '', text, 1)
    
    # First try the standard parser
    result = parse_response_json_like(text)
    
    # If we got entities, return them
    if any(result.get(key, []) for key in ["PER", "ORG", "LOC"]):
        return result
        
    # Otherwise, use aggressive regex extraction
    out = {"PER": [], "ORG": [], "LOC": []}
    for key in ["PER", "ORG", "LOC"]:
        # Match: "PER": [ ... ] or "PER": [
        pattern = rf'"{key}"\s*:\s*\[([^\]]*)'
        m = re.search(pattern, text)
        if m:
            # Extract all quoted strings after the key, even if the list is not closed
            items = re.findall(r'"([^"]+)"', m.group(1))
            out[key] = items
            
    return out



# Create a sample prompt for MLflow logging
def create_sample_prompt(model_name, language, system_prompt_choice=None, use_generic_template=False):
    """
    Create a sample prompt for logging to MLflow based on model type
    """
    # Use a simple sample sentence
    sample_sentence = "John Smith from Microsoft visited Berlin last week."
    few_shots = load_few_shots(language)

    if model_name in ["/gemma-3-4b-it", "/gemma-3-12b-it", "/mistral"] and system_prompt_choice is not None:
        # For Gemma models
        messages = build_gemma_messages(
            sample_sentence, 
            language, 
            system_prompt_choice, 
            few_shots, 
            use_generic_template=use_generic_template
        )
        return {
            "prompt_type": "gemma_chat",
            "messages": messages,
            "system_prompt_choice": system_prompt_choice,
            "use_generic_template": use_generic_template
        }
    else:  # GoLLIE
        prompt = build_gollie_prompt(sample_sentence, language, use_generic_template=use_generic_template)
        return {
            "prompt_type": "gollie_completion",
            "prompt": prompt,
            "use_generic_template": use_generic_template
        }



# ---------- BIO tagging ----------

def get_bio_tags_language_aware(sentence, entities, language="en"):
    """
    Convert sentence and entities to BIO tags with language-specific tokenization.
    For Chinese/Arabic: character-level matching since spaces don't separate words.
    For other languages: word-level matching with space separation.
    """
    if language in ["zh", "ar"]:
        # Character-level approach for Chinese and Arabic
        chars = list(sentence.replace(" ", ""))  # Remove any spaces, work with characters
        tags = ["O"] * len(chars)
        
        for ent_type, mentions in entities.items():
            for mention in mentions:
                mention_clean = mention.replace(" ", "")  # Remove spaces from mention
                mention_chars = list(mention_clean)
                
                # Find character-level matches
                for i in range(len(chars) - len(mention_chars) + 1):
                    if chars[i:i+len(mention_chars)] == mention_chars:
                        # Check if this span is already tagged to avoid conflicts
                        if all(tags[i+j] == "O" for j in range(len(mention_chars))):
                            tags[i] = "B-" + ent_type
                            for j in range(1, len(mention_chars)):
                                tags[i+j] = "I-" + ent_type
                        break  # Mark only the first occurrence
        
        return chars, tags
    else:
        # Word-level approach for space-separated languages
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

def get_bio_tags(sentence, entities):
    """Backward compatibility wrapper - defaults to English behavior"""
    return get_bio_tags_language_aware(sentence, entities, "en")


# ---------- telemetry ----------

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

async def process_batch_chat(session, sentences, max_tokens, model_name, language, system_prompt_choice=None, use_generic_template=False):

    async def single_request(sentence):
        few_shots = load_few_shots(language)
        
        # Use different message construction based on model
        if model_name in ["/gemma-3-4b-it","/gemma-3-12b-it", "/mistral"] and system_prompt_choice is not None:
            # For Gemma, use the specialized message construction with system_prompt_choice
            messages = build_gemma_messages(
                sentence, 
                language, 
                system_prompt_choice, 
                few_shots, 
                use_generic_template=use_generic_template
            )
        else:
            # For Mistral, use the standard message construction
            messages = make_messages_for(sentence, language, few_shots)
    
            
        payload = {
            "model": model_name,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "stop": ["}"]
        }
        async with session.post(CHAT_URL, json=payload) as resp:
            result = await resp.json()
            return result
            
    tasks = [single_request(s) for s in sentences]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return results

async def process_batch_completion(session, sentences, max_tokens, model_name, language, system_prompt_choice=None, use_generic_template=False):
    async def single_request(sentence):
        # Only GoLLIE uses completions API
        prompt = build_gollie_prompt(sentence, language, use_generic_template)
        payload = {
            "model": model_name,
            "prompt": prompt,
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "stop": ["]\n", "\n]", "]"]
        }
        # Use completions URL for GoLLIE
        async with session.post(COMPLETIONS_URL, json=payload, timeout=aiohttp.ClientTimeout(total=60)) as resp:
            if resp.status != 200:
                return {"error": f"HTTP {resp.status}", "text": await resp.text()}
            return await resp.json()
    
    tasks = [single_request(s) for s in sentences]
    return await asyncio.gather(*tasks, return_exceptions=True)

async def process_and_measure(session, prompts, max_tokens, model_name, language, system_prompt_choice=None, use_generic_template=False):
    e0 = read_energy_joules()
    v0 = read_vllm_metrics()
    t0 = time.perf_counter()
    print("Before batch request: ", datetime.datetime.now())

    if model_name in ["/mistral", "/gemma-3-4b-it", "/gemma-3-12b-it"]:
        # Both Mistral and Gemma use chat API with process_batch_chat
        responses = await process_batch_chat(
            session, 
            prompts, 
            max_tokens, 
            model_name, 
            language, 
            system_prompt_choice, 
            use_generic_template
        )
    else:  # GoLLIE
        responses = await process_batch_completion(session, prompts, max_tokens, model_name, language, use_generic_template=use_generic_template)

    t1 = time.perf_counter()
    print("After batch request: ", datetime.datetime.now())
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
        "e2e_latency_mean": e2e_latency_mean,
        "diff_latency": latency - e2e_latency_mean,
        "energy_j": joules,
        "prompt_tokens": prompt_t,
        "generation_tokens": gen_t,
        "total_tokens": total_t,
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
    test_dataset, label_list, batch_size, model_name, max_new_tokens=150, language="de", 
    system_prompt_choice=None, use_generic_template=False
):
    all_gold, all_pred = [], []
    generated_results = []
    batch_telemetry = []

    async with aiohttp.ClientSession() as session:
        for i in tqdm(range(0, len(test_dataset), batch_size)):
            end = min(i + batch_size, len(test_dataset))
            batch = test_dataset.select(range(i, end))
            sentences = []
            gold_batch = []

            for example in batch:
                tokens = example["tokens"]
                ner_tags = example["ner_tags"]
                
                if language in ["zh", "ar"]:
                    # For Chinese/Arabic: join without spaces to preserve character structure
                    sentence = "".join(tokens)
                else:
                    # For other languages: join with spaces
                    sentence = " ".join(tokens)
                    
                gold_tags = [label_list[tag] for tag in ner_tags]
                sentences.append(sentence)
                gold_batch.append(gold_tags)

            try:
                responses, telemetry = await process_and_measure(
                    session, sentences, max_new_tokens, model_name, language, 
                    system_prompt_choice, use_generic_template
                )
                batch_telemetry.append(telemetry)

                for (response, gold_tags, sentence) in zip(responses, gold_batch, sentences):
                    # Extract text from the response:
                    if isinstance(response, Exception):
                        text = ""
                    elif isinstance(response, dict) and "choices" in response and len(response["choices"]) > 0:
                        if model_name in ["/mistral", "/gemma-3-4b-it", "/gemma-3-12b-it"]:
                            # Chat format for both Mistral and Gemma
                            text = response["choices"][0]["message"]["content"] \
                                   if "message" in response["choices"][0] \
                                   else response["choices"][0].get("text", "")
                        else:
                            # Completions format for GoLLIE
                            text = response["choices"][0].get("text", "")
                    else:
                        text = str(response)

                    # Parse the text based on model type
                    if model_name == "/mistral":
                        entities = parse_response_json_like(text)
                    elif model_name in ["/gemma-3-4b-it", "/gemma-3-12b-it"]:
                        # Add debugging for the first few examples
                        if i < batch_size and len(generated_results) < 3:
                            print(f"\nDEBUG Gemma Entity Extraction for example {len(generated_results)}:")
                            print(f"Raw text starts with: {text[:100]}...")
                        entities = robust_gemma_entity_extraction(text)
                        # Print extracted entities for debugging
                        if i < batch_size and len(generated_results) < 3:
                            print(f"Extracted entities: {entities}")
                    else:
                        entities = parse_gollie_output_to_entities(text)

                    # Debug: print entity extraction for Arabic
                    if language == "ar" and entities:
                        print(f"DEBUG: Sentence: {sentence[:50]}...")
                        print(f"DEBUG: Raw output: {text[:100]}...")
                        print(f"DEBUG: Extracted entities: {entities}")

                    _, pred_tags = get_bio_tags_language_aware(sentence, entities, language)
                    
                    # Debug: print BIO tag conversion for Arabic
                    if language == "ar" and entities:
                        print(f"DEBUG: Generated pred_tags length: {len(pred_tags)}")
                        print(f"DEBUG: Gold tags length: {len(gold_tags)}")
                        print(f"DEBUG: First 10 pred_tags: {pred_tags[:10]}")

                    # align predictions with gold tags
                    if len(pred_tags) < len(gold_tags):
                        pred_tags += ["O"] * (len(gold_tags) - len(pred_tags))
                    elif len(pred_tags) > len(gold_tags):
                        pred_tags = pred_tags[:len(gold_tags)]

                    generated_results.append({
                        "sentence": sentence,
                        "raw_output": text,
                        "entities": entities,
                        "gold_tags": gold_tags,
                        "pred_tags": pred_tags
                    })
                    all_gold.append(gold_tags); all_pred.append(pred_tags)
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


def compute_energy_corrected(
    csv_path: str,
    util_col: str = "gpu_util",
    energy_col: str = "energy_consumption",
    timestamp_col: str = "timestamp",
    util_threshold: float = 0.0,   # strictly > 0 regarded as "active"
    energy_in_mJ: bool = True,     # DCGM TOTAL_ENERGY_CONSUMPTION is in millijoules,
    start_timestamp: str = None,   # Optional ISO timestamp to start analysis from
) -> dict:
    """
    Compute energy between the FIRST non-zero GPU util sample and the LAST non-zero GPU util sample.
    When start_timestamp is provided, only consider rows with timestamp >= start_timestamp.
    Any zero-util gaps in the middle are ignored.

    Returns:
        {
          ok, reason?,
          energy_corrected (J),
          latency_corrected (s),
          t_start (iso), t_end (iso),
          start_util, end_util,
          n_rows, n_active
        }
    """
    if not csv_path or not os.path.isfile(csv_path):
        return {"ok": False, "reason": f"CSV missing: {csv_path}"}

    try:
        df = pd.read_csv(
            csv_path,
            parse_dates=[timestamp_col],
            dtype={util_col: "float64", energy_col: "float64"},
            on_bad_lines="skip",
            engine="python",
        )
    except Exception as e:
        return {"ok": False, "reason": f"Failed to read CSV: {e}"}

    if timestamp_col not in df or util_col not in df or energy_col not in df:
        return {"ok": False, "reason": f"CSV lacks required columns: {timestamp_col}, {util_col}, {energy_col}"}

    # Clean & order
    df = (
        df.dropna(subset=[timestamp_col, util_col, energy_col])
          .sort_values(timestamp_col, kind="mergesort")  # stable sort
          .reset_index(drop=True)
    )
    
    # Filter by start timestamp if provided
    if start_timestamp:
        try:
            start_dt = pd.to_datetime(start_timestamp)
            df = df[df[timestamp_col] >= start_dt]
        except Exception as e:
            return {"ok": False, "reason": f"Invalid start_timestamp format: {e}"}
        
    if df.empty:
        return {"ok": False, "reason": "No valid rows after filtering."}

    util = df[util_col].to_numpy()
    active_idx = np.nonzero(util > util_threshold)[0]

    if active_idx.size == 0:
        return {"ok": False, "reason": "GPU util never > 0."}

    start_pos = int(active_idx[0])
    end_pos   = int(active_idx[-1])

    # Indices
    start_idx = start_pos
    end_idx   = end_pos

    # Extract values
    e0 = float(df.at[start_idx, energy_col])
    e1 = float(df.at[end_idx,   energy_col])

    # Convert mJ -> J if needed
    if energy_in_mJ:
        e0 /= 1000.0
        e1 /= 1000.0

    energy_corrected = max(e1 - e0, 0.0)

    t_start = pd.to_datetime(df.at[start_idx, timestamp_col])
    t_end   = pd.to_datetime(df.at[end_idx,   timestamp_col])
    latency_s = max((t_end - t_start).total_seconds(), 0.0)

    return {
        "ok": True,
        "energy_corrected": energy_corrected,
        "latency_corrected": latency_s,
        "t_start": t_start.isoformat(),
        "t_end": t_end.isoformat(),
        "start_util": float(df.at[start_idx, util_col]),
        "end_util": float(df.at[end_idx,   util_col]),
        "n_rows": int(len(df)),
        "n_active": int(active_idx.size),
    }

async def run(args):

    subset = resolve_xtreme_subset(args.language)
    print(f"Loading XTREME subset: {subset}")
    test_ds = load_dataset("google/xtreme", subset, split="test", trust_remote_code=True)
    test_subset = test_ds.select(range(min(args.limit if args.limit else 10000, len(test_ds))))
    labels = test_ds.features["ner_tags"].feature.names
    print(f"Loaded {args.language} dataset with {len(test_subset)} samples and {len(labels)} labels: {labels}")

    artifacts_dir = build_artifacts_dir(args.language, args.batch_size, args.model)
    base_dir = "./inference_eval_artifacts/xtreme"
    out_dir = os.path.join(base_dir, artifacts_dir)
    os.makedirs(os.path.join(out_dir, "generated_responses"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "ner_metrics"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "telemetry"), exist_ok=True)
    

    experiment_name = f"{args.model}_xtreme"

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment_name)
    # ----- MLflow run -----
    # Include timestamp in run name if provided
    timestamp_suffix = ""
    if getattr(args, 'run_start_timestamp', None):
        try:
            # Extract just the time part for the run name to keep it short
            dt = pd.to_datetime(args.run_start_timestamp)
            timestamp_suffix = f"_T{dt.strftime('%H%M%S')}"
        except:
            pass
    
    with mlflow.start_run(run_name=f"xtreme_ner_{args.language}_prompt{args.system_prompt_choice}{timestamp_suffix}_{job_id}"):
        tags = {
            "language": args.language,
            "batch_size": str(args.batch_size),
            "model": args.model,
            "dataset": "xtreme",
            "n_samples" : str(len(test_subset))
        }
        params = {
            "language": args.language,
            "batch_size": args.batch_size,
            "model": args.model,
            "max_new_tokens": args.max_new_tokens,
            "dataset": "xtreme",
            "n_samples" : str(len(test_subset))
        }
        
        # Add Gemma-specific parameters if needed
        if args.model in ["/gemma-3-4b-it", "/gemma-3-12b-it", "/mistral"] and args.system_prompt_choice is not None:
            tags["system_prompt_choice"] = str(args.system_prompt_choice)
            params["system_prompt_choice"] = args.system_prompt_choice
            
        # Add template usage info
        if args.use_generic_template:
            tags["use_generic_template"] = "True"
            params["use_generic_template"] = True
            
        mlflow.set_tags(tags)
        mlflow.log_params(params)
        
        # Create and log a sample prompt to MLflow
        sample_prompt = create_sample_prompt(
            args.model, 
            args.language, 
            args.system_prompt_choice if args.model in ["/gemma-3-4b-it", "/gemma-3-12b-it", "/mistral"] else None,
            use_generic_template=args.use_generic_template
        )
        
        # Save sample prompt to JSON file and log as artifact
        sample_prompt_path = os.path.join(out_dir, "sample_prompt.json")
        with open(sample_prompt_path, "w") as f:
            json.dump(sample_prompt, f, indent=2, ensure_ascii=False)
        mlflow.log_artifact(sample_prompt_path)

        ner_metrics, gen_responses, batch_telemetry = await evaluate_ner_pipeline_xtreme(
            test_subset, labels, batch_size=args.batch_size, model_name=args.model,
            max_new_tokens=args.max_new_tokens, language=args.language,
            system_prompt_choice=args.system_prompt_choice if args.model in ["/gemma-3-4b-it", "/gemma-3-12b-it", "/mistral"]   else None,
            use_generic_template=args.use_generic_template
        )

        responses_path = os.path.join(out_dir, "generated_responses", f"responses_B{args.batch_size}.json")
        ner_metrics_path = os.path.join(out_dir, "ner_metrics", f"ner_metrics_B_{args.batch_size}.json")
        telemetry_path = os.path.join(out_dir, "telemetry", f"telemetry_B_{args.batch_size}.json")
        
        with open(responses_path, "w") as f:
            json.dump(gen_responses, f, indent=2, default=numpy_serializer, ensure_ascii=False)
        with open(ner_metrics_path, "w") as f:
            json.dump(ner_metrics, f, indent=2, default=numpy_serializer, ensure_ascii=False)
        with open(telemetry_path, "w") as f:
            json.dump(batch_telemetry, f, indent=2, default=numpy_serializer, ensure_ascii=False)


        mlflow.log_artifact(responses_path)
        mlflow.log_artifact(ner_metrics_path)
        mlflow.log_artifact(telemetry_path)

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

            mean_metrics_path = os.path.join(out_dir, f"mean_metrics_B_{args.batch_size}.json")
            with open(mean_metrics_path, "w") as f:
                json.dump(mean_metrics, f, indent=2, default=numpy_serializer)
            mlflow.log_artifact(mean_metrics_path)

        time.sleep(10)
        # Get the start timestamp from args if provided, otherwise None (use full CSV)
        run_start_timestamp = getattr(args, 'run_start_timestamp', None)
        
        res = compute_energy_corrected(
            metrics_csv_path,
            start_timestamp=run_start_timestamp
        )
        
        if res.get("ok"):
            mlflow.log_metric("energy_corrected", float(res["energy_corrected"]))
            mlflow.log_metric("latency_corrected", float(res["latency_corrected"]))
            mean_metrics.update({
                "energy_corrected": float(res["energy_corrected"]),
                "latency_corrected": float(res["latency_corrected"])
            })
            mlflow.log_params({
                "energy_window_start": res["t_start"],
                "energy_window_end": res["t_end"],
                "energy_start_util": res["start_util"],
                "energy_end_util": res["end_util"],
                "run_start_timestamp": run_start_timestamp,
            })
        else:
            mlflow.set_tag("energy_window_note", f"skipped: {res.get('reason')}")

        if metrics_csv_path and os.path.isfile(metrics_csv_path):
            mlflow.log_artifact(metrics_csv_path)
        print(f"Completed: F1={ner_metrics['f1']:.4f}, Energy Mean={mean_metrics['mean_energy_j']:.4f}J, Whole Energy={mean_metrics['whole_energy']:.4f}J")

def main():
    parser = argparse.ArgumentParser(description="XTREME NER evaluation with vLLM + energy & MLflow")
    # Get all supported languages from resolve_xtreme_subset
    supported_languages = sorted([
        "ar", "bg", "de", "en", "es", "fr", "el", "hi", "id", 
        "it", "ja", "ko", "nl", "pt", "ru", "th", "tr", "ur", "vi", "zh", "yo"
    ])
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        choices=supported_languages + ["all"],
        help="Language subset from XTREME PAN-X. Use 'all' to run all supported languages."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["/mistral", "/gollie", "/gemma-3-4b-it", "/gemma-3-12b-it"],
        help="Model identifier passed in the request payload."
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
    parser.add_argument(
        "--system-prompt-choice",
        type=int,
        choices=range(1, 9),  # 1-8
        default=1,
        help="Which system prompt to use for Gemma & Mistral (1-8). Only used if model is '/gemma'."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of test examples (default: 10000)."
    )
    parser.add_argument(
        "--run-start-timestamp", 
        type=str,
        default=None,
        help="ISO timestamp marking the start of this run for energy calculation. Format: YYYY-MM-DDTHH:MM:SS.ssssss"
    )
    parser.add_argument(
        "--use-generic-template",
        action="store_true",
        help="Use the generic Template_system_prompt.json instead of language-specific system prompts"
    )
    args = parser.parse_args()
    
    if args.language == "all":
        # Run for all supported languages sequentially
        languages = sorted([
            "ar", "bg", "de", "en", "es", "fr", "el", "hi", "id", 
            "it", "ja", "ko", "nl", "pt", "ru", "th", "tr", "ur", "vi", "zh"
        ])
        for lang in languages:
            try:
                print(f"\n\n===== Running for language: {lang} ({full_lang_name(lang)}) =====\n")
                args.language = lang
                asyncio.run(run(args))
            except Exception as e:
                print(f"Error processing language {lang}: {e}")
                continue
    else:
        # Run for a single language
        asyncio.run(run(args))


if __name__ == "__main__":
    main()
