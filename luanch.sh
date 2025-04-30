#!/bin/bash


# Lab 1
# 0.6B Models on GPUs 1, 2, 3
conda activate vllm-env
CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3-0.6B-Base --max-model-len 4096 --port 9001 

conda activate vllm-env
CUDA_VISIBLE_DEVICES=2 python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3-0.6B-FP8  --max-model-len 4096 --port 9002 

conda activate vllm-env
CUDA_VISIBLE_DEVICES=3 python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3-0.6B     --max-model-len 4096 --port 9003 

# 14B Models on GPU 0

conda activate vllm-env
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3-14B-FP8   --max-model-len 4096 --port 9004 

conda activate vllm-env
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3-14B-Base  --max-model-len 4096 --port 9005 

conda activate vllm-env
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3-14B       --max-model-len 4096 --port 9006 


