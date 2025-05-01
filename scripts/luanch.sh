#!/bin/bash


# Lab 0
# -----------------------------------------------------------------------------
# 0.6B Models on GPUs 1, 2, 3

# 0 -> 3
# 2 -> 3
# 3 -> 0
# 1 -> 2

conda activate vllm-env
CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3-0.6B-Base --max-model-len 4096 --gpu-memory-utilization 0.95 --port 7000

conda activate vllm-env
# MODEL_DIR=$(ls -d ~/.cache/huggingface/hub/models--${MODEL_ID//\//--}/snapshots/* | head -n 1)
CUDA_VISIBLE_DEVICES=2 python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3-0.6B-FP8 --max-model-len 4096 --gpu-memory-utilization 0.95 --port 7001

conda activate vllm-env
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3-0.6B --max-model-len 4096 --gpu-memory-utilization 0.95 --port 7002

# 14B Models on GPU 0 --> 3
conda activate vllm-env
CUDA_VISIBLE_DEVICES=3 python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3-14B-FP8   --max-model-len 4096 --gpu-memory-utilization 0.95 --port 7003


# ---------------------------------------------------- Pending -----------------------------------
conda activate vllm-env
CUDA_VISIBLE_DEVICES=3 python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3-14B-Base  --max-model-len 4096 --port 7004

conda activate vllm-env
CUDA_VISIBLE_DEVICES=3 python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3-14B       --max-model-len 4096 --port 7005
# -----------------------------------------------------------------------------



# Lab 2
# -----------------------------------------------------------------------------
conda activate vllm-env
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3-8B-FP8   --max-model-len 4096 --port 7003
## CANNOT LOAD ON 24G ^^^^^^^^^^^^^^^^^^^^^^^^^^^

# ------------------------------ working -------------------------------------
conda activate vllm-env
CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3-8B  --max-model-len 4096 --port 7004

conda activate vllm-env
CUDA_VISIBLE_DEVICES=2 python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3-8B-Base   --max-model-len 4096 --port 7005

conda activate vllm-env
CUDA_VISIBLE_DEVICES=3 python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3-1.7B-FP8 --max-model-len 4096 --port 7000

conda activate vllm-env
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3-1.7B --max-model-len 4096 --port 7001
# -----------------------------------------------------------------------------


# Lab 1 - VS Code
# -----------------------------------------------------------------------------
conda activate vllm-env
CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3-4B-FP8 --max-model-len 4096 --port 5000

conda activate vllm-env
CUDA_VISIBLE_DEVICES=2 python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3-4B --max-model-len 4096 --port 5001

conda activate vllm-env
CUDA_VISIBLE_DEVICES=3 python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3-4B-Base --max-model-len 4096 --port 5002
# -----------------------------------------------------------------------------


# Lab 3
# -----------------------------------------------------------------------------
conda activate vllm-env
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3-1.7B-Base --max-model-len 4096 --port 5002




# ssh -L [LOCAL_PORT]:[REMOTE_HOST]:[REMOTE_PORT] user@ssh_server
# PF lab 1 -> 0
ssh -L 7000:xuchang-lab0.staff.sydney.edu.au:7000 ivany@xuchang-lab0.staff.sydney.edu.au -p 8222

# PF lab 1 -> 2
ssh -L 7000:xuchang-lab0.staff.sydney.edu.au:7000 ivany@xuchang-lab2.staff.sydney.edu.au -p 8222
ssh -L 7000:xuchang-lab0.staff.sydney.edu.au:7001 ivany@xuchang-lab2.staff.sydney.edu.au -p 8222
ssh -L 7000:xuchang-lab0.staff.sydney.edu.au:7002 ivany@xuchang-lab2.staff.sydney.edu.au -p 8222
ssh -L 7000:xuchang-lab0.staff.sydney.edu.au:7003 ivany@xuchang-lab2.staff.sydney.edu.au -p 8222
ssh -L 7000:xuchang-lab0.staff.sydney.edu.au:7004 ivany@xuchang-lab2.staff.sydney.edu.au -p 8222
ssh -L 7000:xuchang-lab0.staff.sydney.edu.au:7005 ivany@xuchang-lab2.staff.sydney.edu.au -p 8222


python -m LLM-Calibration-Study.simple_evals --model Qwen/Qwen3-235B-A22B-fp8 --benchmark mmlu_pro --conf_mode sampling

