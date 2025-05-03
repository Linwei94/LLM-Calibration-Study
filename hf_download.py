
import os
from transformers import AutoTokenizer, AutoModel
from huggingface_hub import login
login(token=os.environ["HF_TOKEN"])

# Set custom cache directory
# os.environ["HF_HOME"] = "hf_models/"

# List of model names to download
model_names = [
    "Qwen/Qwen2.5-7B-Instruct",
    "google/gemma-2b-it",
    "google/gemma-3-1b-it",
    "google/gemma-3-4b-it",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "mistralai/Mixtral-8x7B-v0.1",
    "mistralai/Mistral-7B-Instruct-v0.1"
]


for model_name in model_names:
    try:
        print(f"Downloading: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
    except:
        print("Failed to download:", model)
print("✅ All models downloaded.")
