
import os
from transformers import AutoTokenizer, AutoModel
from huggingface_hub import login
login(token=os.environ["HF_TOKEN"])

# Set custom cache directory
# os.environ["HF_HOME"] = "hf_models/"

# List of model names to download
model_names = [
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-0.6B-FP8",
    "Qwen/Qwen3-0.6B-Base",
    "Qwen/Qwen3-1.7B-FP8",
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-1.7B-Base",
    "Qwen/Qwen3-4B-FP8",
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-4B-Base",
    "Qwen/Qwen3-8B-FP8",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-8B-Base",
    "Qwen/Qwen3-14B-FP8",
    "Qwen/Qwen3-14B-Base",
    "Qwen/Qwen3-14B",
    "Qwen/Qwen3-30B-A3B-Base",
    "Qwen/Qwen3-30B-A3B-FP8",
    "Qwen/Qwen3-30B-A3B",
    "Qwen/Qwen3-32B-FP8",
    "Qwen/Qwen3-32B",
    "google/gemma-3-27b-it",
    "google/gemma-2-27b-it",
    "google/gemma-2-9b-it",
    "google/gemma-2b-it",
    "google/gemma-3-1b-it",
    "google/gemma-3-4b-it",
    "google/gemma-3-12b-it",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "mistralai/Mixtral-8x7B-v0.1",
    "mistralai/Mistral-7B-Instruct-v0.1",
    "mistralai/Mistral-Small-3.1-24B-Base-2503",
    "mistralai/Mistral-Large-Instruct-2411",
    "mistralai/Mistral-Nemo-Instruct-2407",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
]



for model_name in model_names:
    try:
        print(f"Downloading: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
    except:
        print("Failed to download:", model)
print("✅ All models downloaded.")
