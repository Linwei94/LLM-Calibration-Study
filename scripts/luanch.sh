#!/bin/bash


# Lab 0
# -----------------------------------------------------------------------------
# 0.6B Models on GPUs 1, 2, 3
conda activate vllm-env
CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3-0.6B-Base --max-model-len 4096 --port 7000

conda activate vllm-env
# MODEL_DIR=$(ls -d ~/.cache/huggingface/hub/models--${MODEL_ID//\//--}/snapshots/* | head -n 1)
CUDA_VISIBLE_DEVICES=2 python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3-0.6B-FP8 --max-model-len 4096 --port 7001

conda activate vllm-env
CUDA_VISIBLE_DEVICES=3 python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3-0.6B --max-model-len 4096 --port 7002



# 14B Models on GPU 0
conda activate vllm-env
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3-14B-FP8   --max-model-len 4096 --port 7003

conda activate vllm-env
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3-14B-Base  --max-model-len 4096 --port 7004


conda activate vllm-env
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3-14B       --max-model-len 4096 --port 7005
# -----------------------------------------------------------------------------

# PF
ssh -L [LOCAL_PORT]:[REMOTE_HOST]:[REMOTE_PORT] user@ssh_server
ssh -L 7000:xuchang-lab0.staff.sydney.edu.au:7000 ivany@xuchang-lab0.staff.sydney.edu.au -p 8222


python -m LLM-Calibration-Study.simple_evals --model meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo --benchmark mmlu_pro --conf_mode verbal_linguistic_shared_sampling

