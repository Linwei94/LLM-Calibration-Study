import os

for f in sorted(os.listdir("/home/ivan/LLM-Calibration-Study/cache")):
    if "tmp" not in f and "full" in f and "faulty" not in f:
        cache = f.replace("mmlu_pro_shared_sampling_", "").replace("_full_0", "")
        if "gemma" in cache:
            cache = "google/" + cache
        elif "Mixtral" in cache or "Mistral" in cache:
            cache = "mistralai/" + cache
        elif "Qwen" in cache:
            cache = "Qwen/" + cache
        elif "llama" in cache.lower():
            cache = "meta-llama/" + cache
        print(cache)