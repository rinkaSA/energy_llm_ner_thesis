#!/usr/bin/env python3
import json
import os
import re
import ast
import time
import datetime
import asyncio
import aiohttp
import requests
import numpy as np
import sys
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass
from tqdm import tqdm
from datasets import load_dataset
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
from dotenv import load_dotenv
import argparse
import mlflow
import pandas as pd
from math import isnan
import torch

# Add the serve_vllm directory to Python path to find prompts_in_all_languages module
script_dir = os.path.dirname(os.path.abspath(__file__))
serve_vllm_dir = os.path.join(script_dir, '../../')
sys.path.insert(0, os.path.abspath(serve_vllm_dir))

# ---------------------------
# Load environment variables
# ---------------------------
load_dotenv()

# Environment constants
CHAT_URL = os.getenv("CHAT_URL")
COMPLETIONS_URL = os.getenv("COMPLETIONS_URL")
ENERGY_URL = os.getenv("ENERGY_URL")
VLLM_METRICS_URL = os.getenv("VLLM_METRICS_URL")
ENERGY_METRIC_NAME = "DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
FEW_SHOTS_PATH = os.path.join(serve_vllm_dir, "prompts_in_all_languages/ner_few_shots_masakha.json")
metrics_csv_path = os.environ.get("METRICS_CSV")
job_id = os.getenv("JOB_ID")

print(f"CHAT_URL: {CHAT_URL}")
print(f"JOB_ID: {job_id}")

# Import GoLLIE prompt headers
from prompts_in_all_languages.gollie_prompts import (
    ENGLISH_HEADER, GERMAN_HEADER, CHINESE_HEADER, ARABIC_HEADER, BULGARIAN_HEADER
)

# Target entity types
TARGET_TYPES = {"PER", "ORG", "LOC"}

# Language mappings for MasakhaNER2 dataset
MASAKHANER2_LANGS = {
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

# Class patterns for regex-based entity extraction
CLASS_PATTERNS = {
    "PER": re.compile(r'PER\s*\(\s*mention\s*=\s*(?P<q>["\'])(?P<val>.+?)\1\s*\)'),
    "ORG": re.compile(r'ORG\s*\(\s*mention\s*=\s*(?P<q>["\'])(?P<val>.+?)\1\s*\)'),
    "LOC": re.compile(r'LOC\s*\(\s*mention\s*=\s*(?P<q>["\'])(?P<val>.+?)\1\s*\)'),
}

# Mapping of language codes to headers for GoLLIE
LANG2HEADER = {
    "bam": ENGLISH_HEADER, "ewe": ENGLISH_HEADER, "fon": ENGLISH_HEADER,
    "hau": ENGLISH_HEADER, "ibo": ENGLISH_HEADER, "kin": ENGLISH_HEADER,
    "lug": ENGLISH_HEADER, "luo": ENGLISH_HEADER, "mos": ENGLISH_HEADER,
    "pcm": ENGLISH_HEADER, "sna": ENGLISH_HEADER, "swa": ENGLISH_HEADER,
    "tsn": ENGLISH_HEADER, "twi": ENGLISH_HEADER, "wol": ENGLISH_HEADER,
    "xho": ENGLISH_HEADER, "yor": ENGLISH_HEADER, "zul": ENGLISH_HEADER
}

# Utility functions for serialization
def numpy_serializer(obj):
    """Serialize numpy objects for JSON"""
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Type {obj.__class__.__name__} not serializable")


@dataclass
class ModelRequest:
    """Data class for model request parameters"""
    sentence: str
    language: str
    use_generic_template: bool = False
    system_prompt_choice: Optional[int] = None


@dataclass
class ModelResponse:
    """Data class for model response"""
    text: str
    error: Optional[str] = None
    raw_response: Optional[Dict] = None


@dataclass
class EvaluationResult:
    """Data class for evaluation results"""
    sentence: str
    raw_output: str
    entities: Dict[str, List[str]]
    gold_tags: List[str]
    pred_tags: List[str]


@dataclass
class Metrics:
    """Data class for metrics"""
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    cls_report: Dict = None


@dataclass
class TelemetryMetrics:
    """Data class for telemetry metrics"""
    batch_size: int = 0
    latency_s: float = 0.0
    energy_j: float = 0.0
    prompt_tokens: int = 0
    generation_tokens: int = 0
    total_tokens: int = 0
    e2e_latency_mean: float = 0.0
    ttft_mean: float = 0.0
    time_per_token_mean: float = 0.0
    prefill_total_s: float = 0.0
    inference_total_s: float = 0.0
    decode_total_s: float = 0.0


class PromptStrategy(ABC):
    """Abstract base class for prompt building strategies"""
    
    @abstractmethod
    async def build_prompt(self, request: ModelRequest) -> Dict:
        """Build prompt for the model"""
        pass
    
    @abstractmethod
    async def process_batch(self, session: aiohttp.ClientSession, requests: List[ModelRequest], 
                           max_tokens: int, model_name: str) -> List[ModelResponse]:
        """Process a batch of requests"""
        pass


class ChatPromptStrategy(PromptStrategy):
    """Chat-based prompt strategy for models like Mistral and Gemma"""
    
    def __init__(self):
        self.few_shots_cache = {}
    
    def get_system_template(self, language: str) -> str:
        """Get system prompt template"""
        return (
            "You are a Named Entity Recognition (NER) annotator for the {language} language.\n"
            "- Input: a single {language} sentence (do not translate).\n"
            "- Output: only a single, valid JSON object with three arrays: `PER`, `ORG`, `LOC`.\n"
            "- Do not output any extra text, explanations, or markdown.\n"
            "- JSON must be parseable by json.loads.\n"
            "- Extract only entities that appear verbatim in the input sentence (no hallucinations).\n"
            "- Keep entity surface forms exactly as they appear; do not normalize or translate."
        ).format(language=self._get_language_name(language))
    
    def _get_language_name(self, language: str) -> str:
        """Get full language name from code"""
        return MASAKHANER2_LANGS.get(language, language)
    
    def build_system_message(self, language: str) -> Dict:
        """Build system message for chat models"""
        return {
            "role": "system",
            "content": self.get_system_template(language)
        }
    
    def load_gemma_system_template(self, language: str, 
                                  system_prompt_choice: int,
                                  use_generic_template: bool = False) -> str:
        """Load a system prompt template for Gemma from JSON file by choice number"""
        # Determine template path based on whether to use generic template
        if use_generic_template:
            template_path = os.path.join(serve_vllm_dir, 
                                      "prompts_in_all_languages/Template_system_prompt.json")
        else:
            lang_name = self._get_language_name(language)
            template_path = os.path.join(serve_vllm_dir, 
                                      f"prompts_in_all_languages/{lang_name}_system_prompt.json")
            
            # Fallback to Template for languages without specific templates
            if not os.path.exists(template_path):
                template_path = os.path.join(serve_vllm_dir, 
                                          "prompts_in_all_languages/Template_system_prompt.json")
                use_generic_template = True
        
        # Load template data
        with open(template_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        if str(system_prompt_choice) not in data:
            raise ValueError(f"No system prompt found for choice: {system_prompt_choice}")
        
        # Get prompt from template
        prompt = data[str(system_prompt_choice)]
        if isinstance(prompt, list):
            prompt = "\n".join(prompt)
        
        # Replace language placeholder if using generic template
        if use_generic_template:
            prompt = prompt.replace("{language}", self._get_language_name(language))
        
        return prompt
    
    def load_few_shots(self, language: str) -> List[Dict]:
        """Load few-shot examples for a language"""
        # Use cached few-shots if available
        if language in self.few_shots_cache:
            return self.few_shots_cache[language]
        
        # Load few-shots from file
        with open(FEW_SHOTS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Use English as fallback if language not found
        if language not in data:
            print(f"Warning: No few-shot examples for {language}, using English")
            few_shots = data.get("en", [])
        else:
            few_shots = data[language]
        
        # Cache and return
        self.few_shots_cache[language] = few_shots
        return few_shots
    
    def build_gemma_messages(self, request: ModelRequest) -> List[Dict]:
        """Build messages for Gemma models"""
        # Get system prompt based on choice
        sys_msg = self.load_gemma_system_template(
            language=request.language,
            system_prompt_choice=request.system_prompt_choice or 1,
            use_generic_template=request.use_generic_template
        )
        
        # Create system message
        messages = [{"role": "system", "content": sys_msg}]
        
        # Set number of few-shot examples based on system prompt choice
        choice = request.system_prompt_choice or 1
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
        n_pairs = pairs_for_choice.get(choice, 0)
        
        # Add few-shot examples if needed
        if n_pairs > 0:
            few_shots = self.load_few_shots(request.language)
            # Take only the requested number of example pairs
            for i in range(min(n_pairs * 2, len(few_shots))):
                messages.append(few_shots[i])
        
        # Add the user query with the sentence
        messages.append({"role": "user", "content": request.sentence})
        
        return messages
    
    def make_mistral_messages(self, request: ModelRequest) -> List[Dict]:
        """Build messages for Mistral model"""
        system_msg = self.build_system_message(request.language)
        few_shots = self.load_few_shots(request.language)
        
        return [system_msg] + few_shots + [
            {"role": "user", "content": f"Sentence: {request.sentence}"}
        ]
    
    async def build_prompt(self, request: ModelRequest) -> Dict:
        """Build prompt for chat models"""
        if request.system_prompt_choice is not None:  # Gemma
            messages = self.build_gemma_messages(request)
        else:  # Mistral
            messages = self.make_mistral_messages(request)
        
        return {
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": 150,  # Default max tokens
            "stop": ["}"]
        }
    
    async def process_batch(self, session: aiohttp.ClientSession, 
                           requests: List[ModelRequest],
                           max_tokens: int, model_name: str) -> List[ModelResponse]:
        """Process a batch of requests for chat models"""
        
        async def single_request(request):
            try:
                # Build payload
                payload = await self.build_prompt(request)
                payload["model"] = model_name
                payload["max_tokens"] = max_tokens
                
                # Send request
                async with session.post(CHAT_URL, json=payload) as resp:
                    if resp.status != 200:
                        return ModelResponse(text="", error=f"HTTP {resp.status}")
                    
                    result = await resp.json()
                    
                    # Extract text from the response
                    if "choices" in result and len(result["choices"]) > 0:
                        text = result["choices"][0]["message"]["content"] \
                               if "message" in result["choices"][0] \
                               else result["choices"][0].get("text", "")
                    else:
                        text = ""
                    
                    return ModelResponse(text=text, raw_response=result)
            except Exception as e:
                return ModelResponse(text="", error=str(e))
        
        tasks = [single_request(req) for req in requests]
        return await asyncio.gather(*tasks)


class CompletionPromptStrategy(PromptStrategy):
    """Completion-based prompt strategy for models like GoLLIE"""
    
    def build_gollie_prompt(self, request: ModelRequest) -> str:
        """Build prompt for GoLLIE model"""
        # Use language-specific header if available and not using generic template
        if request.language in LANG2HEADER and not request.use_generic_template:
            header = LANG2HEADER[request.language]
        else:
            # For languages without specific headers, use English header with language name injected
            template_header = ENGLISH_HEADER
            lang_name = MASAKHANER2_LANGS.get(request.language, request.language)
            header = template_header.replace("English language", f"{lang_name} language")
        
        # Format the prompt
        text_literal = json.dumps(request.sentence, ensure_ascii=False)
        lang_name = MASAKHANER2_LANGS.get(request.language, request.language)
        return header + f"\ntext = {text_literal}\n# Annotate entities in the {lang_name} language.\nresult = [\n"
    
    async def build_prompt(self, request: ModelRequest) -> Dict:
        """Build prompt for completion models"""
        prompt = self.build_gollie_prompt(request)
        
        return {
            "prompt": prompt,
            "temperature": 0.0,
            "max_tokens": 150,  # Default max tokens
            "stop": ["]\n", "\n]", "]"]
        }
    
    async def process_batch(self, session: aiohttp.ClientSession, 
                           requests: List[ModelRequest],
                           max_tokens: int, model_name: str) -> List[ModelResponse]:
        """Process a batch of requests for completion models"""
        
        async def single_request(request):
            try:
                # Build payload
                payload = await self.build_prompt(request)
                payload["model"] = model_name
                payload["max_tokens"] = max_tokens
                
                # Send request
                async with session.post(COMPLETIONS_URL, json=payload, 
                                       timeout=aiohttp.ClientTimeout(total=60)) as resp:
                    if resp.status != 200:
                        return ModelResponse(text="", error=f"HTTP {resp.status}")
                    
                    result = await resp.json()
                    
                    # Extract text from the response
                    if "choices" in result and len(result["choices"]) > 0:
                        text = result["choices"][0].get("text", "")
                    else:
                        text = ""
                    
                    return ModelResponse(text=text, raw_response=result)
            except Exception as e:
                return ModelResponse(text="", error=str(e))
        
        tasks = [single_request(req) for req in requests]
        return await asyncio.gather(*tasks)


class EntityExtractor(ABC):
    """Abstract base class for entity extraction strategies"""
    
    @abstractmethod
    def extract_entities(self, response: ModelResponse) -> Dict[str, List[str]]:
        """Extract entities from model response"""
        pass


class ChatEntityExtractor(EntityExtractor):
    """Entity extractor for chat models"""
    
    def extract_entities(self, response: ModelResponse) -> Dict[str, List[str]]:
        """Extract entities from chat model response"""
        if response.error or not response.text:
            return {"PER": [], "ORG": [], "LOC": []}
        
        return self.parse_response_json_like(response.text)
    
    def parse_response_json_like(self, response_text: str) -> Dict[str, List[str]]:
        """
        Robustly parse a JSON-like dict out of possibly-broken model output,
        normalizing keys to upper-case.
        """
        cleaned = re.sub(r"^\s*\d+\s*=\s*", "", response_text, flags=re.MULTILINE)
        cleaned = cleaned.strip().strip("'\"")
        
        # Extract JSON block
        start = cleaned.find("{")
        if start < 0:
            return {"PER": [], "ORG": [], "LOC": []}
            
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
        
        # Get JSON block
        if end_idx is not None:
            block = cleaned[start:end_idx + 1]
        else:
            block = cleaned[start:] + "}" * depth
        
        # Clean up JSON block
        block = re.sub(r",\s*}", "}", block)
        block = block.replace("'", '"')
        
        # Try to parse JSON
        parsed = {}
        for loader in (json.loads, lambda s: ast.literal_eval(s)):
            try:
                parsed = loader(block)
                if isinstance(parsed, dict):
                    break
            except Exception:
                continue
        
        # Normalize keys and values
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
        
        # Ensure all entity types are present
        for key in ["PER", "ORG", "LOC"]:
            if key not in result:
                result[key] = []
        
        return result


class GollieEntityExtractor(EntityExtractor):
    """Entity extractor for GoLLIE model"""
    
    def extract_entities(self, response: ModelResponse) -> Dict[str, List[str]]:
        """Extract entities from GoLLIE model response"""
        if response.error or not response.text:
            return {"PER": [], "ORG": [], "LOC": []}
        
        return self.parse_gollie_output_to_entities(response.text)
    
    def parse_gollie_output_to_entities(self, text: str) -> Dict[str, List[str]]:
        """Parse GoLLIE output to entities"""
        if "]" in text:
            text = text.split("]", 1)[0]
        
        out = {"PER": [], "ORG": [], "LOC": []}
        for k, pat in CLASS_PATTERNS.items():
            vals = [m.group("val").strip() for m in pat.finditer(text)]
            # dedup preserve order
            seen = set()
            dedup = []
            for v in vals:
                if v not in seen:
                    seen.add(v)
                    dedup.append(v)
            out[k] = dedup
            
        return out


class MetricsCollector:
    """Class for collecting metrics"""
    
    def read_energy_joules(self) -> float:
        """Read DCGM energy metric; returns joules (DCGM exposes millijoules)"""
        try:
            r = requests.get(ENERGY_URL, timeout=1.0).text
            for line in r.splitlines():
                if line.startswith(ENERGY_METRIC_NAME):
                    return float(line.split()[-1]) / 1000.0
            raise RuntimeError(f"{ENERGY_METRIC_NAME} not found in /metrics")
        except Exception:
            return 0.0
    
    def read_vllm_metrics(self) -> Dict:
        """Read vLLM metrics"""
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
        except Exception as e:
            print(f"Error reading vLLM metrics: {e}")
            return {k: 0.0 for k in [
                "prompt_tokens_total", "generation_tokens_total", "e2e_latency_sum",
                "e2e_latency_count", "time_to_first_sum", "time_to_first_count",
                "time_per_token_sum", "time_per_token_count", "request_prefill_time_sum",
                "request_inference_time_sum", "request_decode_time_sum"
            ]}
    
    async def collect_metrics(self, session: aiohttp.ClientSession, 
                             requests: List[ModelRequest],
                             max_tokens: int, model_name: str,
                             prompt_strategy: PromptStrategy) -> Tuple[List[ModelResponse], TelemetryMetrics]:
        """Collect metrics while processing requests"""
        e0 = self.read_energy_joules()
        v0 = self.read_vllm_metrics()
        t0 = time.perf_counter()
        print("Before batch request: ", datetime.datetime.now())
        
        # Process batch
        responses = await prompt_strategy.process_batch(
            session, requests, max_tokens, model_name
        )
        
        t1 = time.perf_counter()
        print("After batch request: ", datetime.datetime.now())
        e1 = self.read_energy_joules()
        v1 = self.read_vllm_metrics()
        
        # Calculate metrics
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
        
        total_processing_time = max(e2e_latency_mean * len(requests), 1e-9)
        joules_prefill = joules * (prefill_total / total_processing_time) if total_processing_time > 0 else 0.0
        joules_inference = joules * (inference_total / total_processing_time) if total_processing_time > 0 else 0.0
        joules_decode = joules * (decode_total / total_processing_time) if total_processing_time > 0 else 0.0
        
        metrics = TelemetryMetrics(
            batch_size=len(requests),
            latency_s=latency,
            energy_j=joules,
            prompt_tokens=prompt_t,
            generation_tokens=gen_t,
            total_tokens=total_t,
            e2e_latency_mean=e2e_latency_mean,
            ttft_mean=ttft_mean,
            time_per_token_mean=time_per_token_mean,
            prefill_total_s=prefill_total,
            inference_total_s=inference_total,
            decode_total_s=decode_total
        )
        
        return responses, metrics


class NERModelEvaluator(ABC):
    """Abstract base class for NER model evaluation"""
    
    def __init__(self, model_name: str, batch_size: int, max_new_tokens: int = 150):
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        
        # Initialize strategies based on model type
        if model_name in ["/mistral", "/gemma-3-4b-it"]:
            self.prompt_strategy = ChatPromptStrategy()
            self.entity_extractor = ChatEntityExtractor()
        else:  # GoLLIE
            self.prompt_strategy = CompletionPromptStrategy()
            self.entity_extractor = GollieEntityExtractor()
        
        # Initialize metrics collector
        self.metrics_collector = MetricsCollector()
    
    @abstractmethod
    def load_dataset(self, language: str, limit: int = None):
        """Load dataset for the given language"""
        pass
    
    def get_bio_tags(self, sentence: str, entities: Dict[str, List[str]]) -> Tuple[List[str], List[str]]:
        """Convert sentence and entities to BIO tags"""
        tokens = sentence.split()
        tags = ["O"] * len(tokens)
        
        for ent_type, mentions in entities.items():
            for mention in mentions:
                mention_tokens = mention.split()
                for i in range(len(tokens) - len(mention_tokens) + 1):
                    if tokens[i:i+len(mention_tokens)] == mention_tokens:
                        # Check if this span is already tagged to avoid conflicts
                        if all(tags[i+j] == "O" for j in range(len(mention_tokens))):
                            tags[i] = f"B-{ent_type}"
                            for j in range(1, len(mention_tokens)):
                                tags[i+j] = f"I-{ent_type}"
                        break
        
        return tokens, tags
    
    def project_to_targets(self, tags: List[str], target_types: set = TARGET_TYPES) -> List[str]:
        """Keep only BIO tags whose entity type is in target_types; map others to 'O'"""
        out = []
        for t in tags:
            if t == "O":
                out.append(t)
            else:
                prefix, typ = t.split("-", 1)
                if typ in target_types:
                    out.append(t)
                else:
                    out.append("O")
        return out
    
    def compute_energy_corrected(self, csv_path: str, start_timestamp: str = None) -> Dict:
        """Compute energy between the FIRST and LAST non-zero GPU util samples"""
        if not csv_path or not os.path.isfile(csv_path):
            return {"ok": False, "reason": f"CSV missing: {csv_path}"}
        
        try:
            df = pd.read_csv(
                csv_path,
                parse_dates=["timestamp"],
                dtype={"gpu_util": "float64", "energy_consumption": "float64"},
                on_bad_lines="skip",
                engine="python",
            )
        except Exception as e:
            return {"ok": False, "reason": f"Failed to read CSV: {e}"}
        
        if "timestamp" not in df or "gpu_util" not in df or "energy_consumption" not in df:
            return {"ok": False, "reason": f"CSV lacks required columns: timestamp, gpu_util, energy_consumption"}
        
        # Clean & order
        df = (
            df.dropna(subset=["timestamp", "gpu_util", "energy_consumption"])
              .sort_values("timestamp", kind="mergesort")  # stable sort
              .reset_index(drop=True)
        )
        
        # Filter by start timestamp if provided
        if start_timestamp:
            try:
                start_dt = pd.to_datetime(start_timestamp)
                df = df[df["timestamp"] >= start_dt]
            except Exception as e:
                return {"ok": False, "reason": f"Invalid start_timestamp format: {e}"}
        
        if df.empty:
            return {"ok": False, "reason": "No valid rows after filtering."}
        
        util = df["gpu_util"].to_numpy()
        active_idx = np.nonzero(util > 0.0)[0]
        
        if active_idx.size == 0:
            return {"ok": False, "reason": "GPU util never > 0."}
        
        start_pos = int(active_idx[0])
        end_pos = int(active_idx[-1])
        
        # Extract values
        e0_mj = float(df.iloc[start_pos]["energy_consumption"])
        e1_mj = float(df.iloc[end_pos]["energy_consumption"])
        
        # Convert mJ -> J
        energy_corrected = max((e1_mj - e0_mj) / 1000.0, 0.0)
        
        t_start = pd.to_datetime(df.iloc[start_pos]["timestamp"])
        t_end = pd.to_datetime(df.iloc[end_pos]["timestamp"])
        latency_s = max((t_end - t_start).total_seconds(), 0.0)
        
        return {
            "ok": True,
            "energy_corrected": energy_corrected,
            "latency_corrected": latency_s,
            "t_start": t_start.isoformat(),
            "t_end": t_end.isoformat(),
            "start_util": float(df.iloc[start_pos]["gpu_util"]),
            "end_util": float(df.iloc[end_pos]["gpu_util"]),
            "n_rows": int(len(df)),
            "n_active": int(active_idx.size),
        }
    
    def build_artifacts_dir(self, language: str) -> str:
        """Build artifacts directory name"""
        safe_model = re.sub(r"[^a-zA-Z0-9_.-]+", "_", self.model_name)
        return f"masakha_{language}_B{self.batch_size}_{safe_model}_{job_id}"
    
    def create_sample_prompt(self, language: str, system_prompt_choice: int = None,
                           use_generic_template: bool = False) -> Dict:
        """Create a sample prompt for logging to MLflow"""
        # Use a simple sample sentence
        sample_sentence = "John Smith from Microsoft visited Berlin last week."
        request = ModelRequest(
            sentence=sample_sentence,
            language=language,
            system_prompt_choice=system_prompt_choice,
            use_generic_template=use_generic_template
        )
        
        if self.model_name == "/gemma-3-4b-it" and system_prompt_choice is not None:
            # For Gemma models
            strategy = ChatPromptStrategy()
            messages = strategy.build_gemma_messages(request)
            return {
                "prompt_type": "gemma_chat",
                "messages": messages,
                "system_prompt_choice": system_prompt_choice,
                "use_generic_template": use_generic_template
            }
        elif self.model_name == "/mistral":
            # For Mistral models
            strategy = ChatPromptStrategy()
            messages = strategy.make_mistral_messages(request)
            return {
                "prompt_type": "mistral_chat",
                "messages": messages,
                "use_generic_template": use_generic_template
            }
        else:  # GoLLIE
            strategy = CompletionPromptStrategy()
            prompt = strategy.build_gollie_prompt(request)
            return {
                "prompt_type": "gollie_completion",
                "prompt": prompt,
                "use_generic_template": use_generic_template
            }
    
    async def evaluate(self, language: str, system_prompt_choice: int = None,
                     use_generic_template: bool = False, limit: int = None) -> Tuple[Metrics, List[EvaluationResult], List[TelemetryMetrics]]:
        """Evaluate the model on the given language"""
        # Load dataset
        dataset, label_list = self.load_dataset(language, limit)
        
        all_gold = []
        all_pred = []
        evaluation_results = []
        batch_telemetry = []
        
        async with aiohttp.ClientSession() as session:
            for i in tqdm(range(0, len(dataset), self.batch_size)):
                # Get batch
                end = min(i + self.batch_size, len(dataset))
                batch = dataset.select(range(i, end))
                
                # Prepare sentences and gold tags
                sentences = []
                gold_batch = []
                
                for example in batch:
                    tokens = example["tokens"]
                    ner_tags = example["ner_tags"]
                    
                    # Join tokens to form sentence
                    sentence = " ".join(tokens)
                    
                    # Get gold tags
                    gold_tags = [label_list[tag] for tag in ner_tags]
                    gold_tags = self.project_to_targets(gold_tags)
                    
                    sentences.append(sentence)
                    gold_batch.append(gold_tags)
                
                try:
                    # Prepare requests
                    requests = [
                        ModelRequest(
                            sentence=sentence,
                            language=language,
                            system_prompt_choice=system_prompt_choice,
                            use_generic_template=use_generic_template
                        )
                        for sentence in sentences
                    ]
                    
                    # Process batch and collect metrics
                    responses, metrics = await self.metrics_collector.collect_metrics(
                        session, requests, self.max_new_tokens, 
                        self.model_name, self.prompt_strategy
                    )
                    
                    batch_telemetry.append(metrics)
                    
                    # Process responses
                    for response, gold_tags, sentence in zip(responses, gold_batch, sentences):
                        # Extract entities
                        entities = self.entity_extractor.extract_entities(response)
                        
                        # Convert to BIO tags
                        _, pred_tags = self.get_bio_tags(sentence, entities)
                        
                        # Align predictions with gold tags
                        if len(pred_tags) < len(gold_tags):
                            pred_tags += ["O"] * (len(gold_tags) - len(pred_tags))
                        elif len(pred_tags) > len(gold_tags):
                            pred_tags = pred_tags[:len(gold_tags)]
                        
                        # Add to results
                        evaluation_results.append(EvaluationResult(
                            sentence=sentence,
                            raw_output=response.text,
                            entities=entities,
                            gold_tags=gold_tags,
                            pred_tags=pred_tags
                        ))
                        
                        # Add to metrics calculation
                        all_gold.append(gold_tags)
                        all_pred.append(pred_tags)
                except Exception as e:
                    print(f"Batch {i//self.batch_size} failed: {e}")
                    for gold_tags in gold_batch:
                        all_gold.append(gold_tags)
                        all_pred.append(["O"] * len(gold_tags))
        
        # Calculate metrics
        if len(all_gold) > 0 and len(all_pred) > 0:
            metrics = Metrics(
                precision=precision_score(all_gold, all_pred),
                recall=recall_score(all_gold, all_pred),
                f1=f1_score(all_gold, all_pred),
                cls_report=classification_report(all_gold, all_pred, output_dict=True)
            )
        else:
            metrics = Metrics()
        
        return metrics, evaluation_results, batch_telemetry
    
    async def run_evaluation(self, language: str, system_prompt_choice: int = None,
                          use_generic_template: bool = False, limit: int = None,
                          run_start_timestamp: str = None):
        """Run evaluation and log results to MLflow"""
        # Build artifacts directory
        artifacts_dir = self.build_artifacts_dir(language)
        base_dir = "./inference_eval_artifacts/masakha"
        out_dir = os.path.join(base_dir, artifacts_dir)
        os.makedirs(os.path.join(out_dir, "generated_responses"), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "ner_metrics"), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "telemetry"), exist_ok=True)
        
        # Set up MLflow
        experiment_name = f"{self.model_name}_masakha"
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(experiment_name)
        
        # Include timestamp in run name if provided
        timestamp_suffix = ""
        if run_start_timestamp:
            try:
                # Extract just the time part for the run name to keep it short
                dt = pd.to_datetime(run_start_timestamp)
                timestamp_suffix = f"_T{dt.strftime('%H%M%S')}"
            except:
                pass
        
        # Run evaluation
        with mlflow.start_run(run_name=f"masakha_ner_{language}{timestamp_suffix}_{job_id}"):
            # Set tags and parameters
            tags = {
                "language": language,
                "batch_size": str(self.batch_size),
                "model": self.model_name,
                "dataset": "masakhaner2",
            }
            params = {
                "language": language,
                "batch_size": self.batch_size,
                "model": self.model_name,
                "max_new_tokens": self.max_new_tokens,
                "dataset": "masakhaner2",
            }
            
            # Add Gemma-specific parameters if needed
            if self.model_name == "/gemma-3-4b-it" and system_prompt_choice is not None:
                tags["system_prompt_choice"] = str(system_prompt_choice)
                params["system_prompt_choice"] = system_prompt_choice
            
            # Add template usage info
            if use_generic_template:
                tags["use_generic_template"] = "True"
                params["use_generic_template"] = True
            
            mlflow.set_tags(tags)
            mlflow.log_params(params)
            
            # Create and log a sample prompt to MLflow
            sample_prompt = self.create_sample_prompt(
                language, system_prompt_choice, use_generic_template
            )
            
            # Save sample prompt to JSON file and log as artifact
            sample_prompt_path = os.path.join(out_dir, "sample_prompt.json")
            with open(sample_prompt_path, "w") as f:
                json.dump(sample_prompt, f, indent=2, ensure_ascii=False)
            mlflow.log_artifact(sample_prompt_path)
            
            # Run evaluation
            ner_metrics, evaluation_results, batch_telemetry = await self.evaluate(
                language, system_prompt_choice, use_generic_template, limit
            )
            
            # Save results
            responses_path = os.path.join(out_dir, "generated_responses", f"responses_B{self.batch_size}.json")
            ner_metrics_path = os.path.join(out_dir, "ner_metrics", f"ner_metrics_B_{self.batch_size}.json")
            telemetry_path = os.path.join(out_dir, "telemetry", f"telemetry_B_{self.batch_size}.json")
            
            with open(responses_path, "w") as f:
                json.dump([vars(r) for r in evaluation_results], f, indent=2, default=numpy_serializer, ensure_ascii=False)
            
            with open(ner_metrics_path, "w") as f:
                json.dump(vars(ner_metrics), f, indent=2, default=numpy_serializer, ensure_ascii=False)
            
            with open(telemetry_path, "w") as f:
                json.dump([vars(t) for t in batch_telemetry], f, indent=2, default=numpy_serializer, ensure_ascii=False)
            
            # Log artifacts
            mlflow.log_artifact(responses_path)
            mlflow.log_artifact(ner_metrics_path)
            mlflow.log_artifact(telemetry_path)
            
            # Log metrics
            mlflow.log_metric("precision", float(ner_metrics.precision))
            mlflow.log_metric("recall", float(ner_metrics.recall))
            mlflow.log_metric("f1", float(ner_metrics.f1))
            
            # Calculate and log mean metrics
            if batch_telemetry:
                mean_metrics = {}
                for field in vars(batch_telemetry[0]):
                    values = [getattr(t, field) for t in batch_telemetry]
                    if all(isinstance(v, (int, float)) for v in values):
                        mean_metrics[f"mean_{field}"] = float(np.mean(values))
                
                # Add NER metrics
                whole_energy = float(np.sum([t.energy_j for t in batch_telemetry]))
                mean_metrics.update({
                    "ner_precision": ner_metrics.precision,
                    "ner_recall": ner_metrics.recall,
                    "ner_f1": ner_metrics.f1,
                    "whole_energy": whole_energy
                })
                
                # Log mean metrics
                mlflow.log_metrics(mean_metrics)
                
                # Save mean metrics to file
                mean_metrics_path = os.path.join(out_dir, f"mean_metrics_B_{self.batch_size}.json")
                with open(mean_metrics_path, "w") as f:
                    json.dump(mean_metrics, f, indent=2, default=numpy_serializer)
                mlflow.log_artifact(mean_metrics_path)
            
            # Wait for metrics to be written
            time.sleep(10)
            
            # Compute energy corrected
            res = self.compute_energy_corrected(
                metrics_csv_path,
                start_timestamp=run_start_timestamp
            )
            
            # Log energy metrics
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
            
            # Always log the CSV so you can inspect it next to the run
            if metrics_csv_path and os.path.isfile(metrics_csv_path):
                mlflow.log_artifact(metrics_csv_path)
            
            # Print completion message
            print(f"Completed: F1={ner_metrics.f1:.4f}, Energy Mean={mean_metrics.get('mean_energy_j', 0.0):.4f}J, Whole Energy={mean_metrics.get('whole_energy', 0.0):.4f}J")


class MasakhaEvaluator(NERModelEvaluator):
    """Class for evaluating NER models on MasakhaNER2 dataset"""
    
    def load_dataset(self, language: str, limit: int = None):
        """Load MasakhaNER2 dataset"""
        test_ds = load_dataset("masakhane/masakhaner2", language, split="test")
        labels = test_ds.features["ner_tags"].feature.names
        
        print(f"Loaded MasakhaNER2 {language} dataset with {len(test_ds)} samples and {len(labels)} labels: {labels}")
        
        # Apply limit if specified
        if limit:
            test_ds = test_ds.select(range(min(limit, len(test_ds))))
            print(f"Limited to {len(test_ds)} samples")
        
        return test_ds, labels


async def main():
    """Main function for running the evaluation"""
    parser = argparse.ArgumentParser(description="MasakhaNER2 evaluation with vLLM + energy & MLflow (OOP version)")
    parser.add_argument(
        "--language",
        type=str,
        required=True,
        choices=list(MASAKHANER2_LANGS.keys()),
        help="Language code from MasakhaNER2 dataset"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["/mistral", "/gollie", "/gemma-3-4b-it"],
        help="Model identifier ('/mistral', '/gollie', or '/gemma-3-4b-it')"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Fixed batch size to use (default: 128)"
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=150,
        help="Max tokens to generate per request (default: 150)"
    )
    parser.add_argument(
        "--system-prompt-choice",
        type=int,
        choices=range(1, 8),  # 1-8
        default=1,
        help="Which system prompt to use for Gemma (1-8). Only used if model is '/gemma-3-4b-it'"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Optional cap on number of test examples (default: 1000)"
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
    parser.add_argument(
        "--all-languages",
        action="store_true",
        help="Run evaluation for all supported languages"
    )
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = MasakhaEvaluator(
        model_name=args.model,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens
    )
    
    if args.all_languages:
        # Run for all languages sequentially
        for lang in MASAKHANER2_LANGS.keys():
            try:
                print(f"\n\n===== Running for language: {lang} ({MASAKHANER2_LANGS[lang]}) =====\n")
                await evaluator.run_evaluation(
                    language=lang,
                    system_prompt_choice=args.system_prompt_choice if args.model == "/gemma-3-4b-it" else None,
                    use_generic_template=args.use_generic_template,
                    limit=args.limit,
                    run_start_timestamp=args.run_start_timestamp
                )
            except Exception as e:
                print(f"Error processing language {lang}: {e}")
                continue
    else:
        # Run for a single language
        await evaluator.run_evaluation(
            language=args.language,
            system_prompt_choice=args.system_prompt_choice if args.model == "/gemma-3-4b-it" else None,
            use_generic_template=args.use_generic_template,
            limit=args.limit,
            run_start_timestamp=args.run_start_timestamp
        )


if __name__ == "__main__":
    asyncio.run(main())
