from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
import os

model_path = "mistralai/Mistral-7B-v0.1"
quant_path = "/data/horse/ws/irve354e-energy_llm_ner/energy_ner_llm/qlora-ner/serve_vllm/models/base/Mistral-7B-AWQ"
quant_config = {
    "w_bit": 4,
    "q_group_size": 128,
    "zero_point": True,
    "version": "GEMM"
}

model = AutoAWQForCausalLM.from_pretrained(
    model_path,
    low_cpu_mem_usage=True,
    use_cache=False
)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


model.quantize(tokenizer, quant_config=quant_config)


os.makedirs(quant_path, exist_ok=True)
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)

print(f"Quantized model saved at {quant_path}")