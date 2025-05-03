#!/bin/bash


# Lab 0
# -----------------------------------------------------------------------------
# 0.6B Models on GPUs 1, 2, 3

# 0 -> 3
# 2 -> 3
# 3 -> 0
# 1 -> 2

conda activate vllm-env
CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3-0.6B-Base --max-model-len 4096 --port 7000
ssh -L 7000:xuchang-lab0.staff.sydney.edu.au:7000 ivany@xuchang-lab0.staff.sydney.edu.au -p 8222


conda activate vllm-env
CUDA_VISIBLE_DEVICES=2 python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3-0.6B-FP8 --max-model-len 4096 --port 7001
ssh -L 7001:xuchang-lab0.staff.sydney.edu.au:7001 ivany@xuchang-lab0.staff.sydney.edu.au -p 8222


conda activate vllm-env
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3-0.6B --max-model-len 4096 --port 7002
ssh -L 7002:xuchang-lab0.staff.sydney.edu.au:7002 ivany@xuchang-lab0.staff.sydney.edu.au -p 8222

conda activate vllm-env
CUDA_VISIBLE_DEVICES=3 python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3-14B   --max-model-len 4096 --port 7005
ssh -L 7005:xuchang-lab0.staff.sydney.edu.au:7005 ivany@xuchang-lab0.staff.sydney.edu.au -p 8222


# -----------------------------------------------------------------------------



# Lab 2
# -----------------------------------------------------------------------------
conda activate vllm-env
CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3-8B  --max-model-len 4096 --port 7004 
ssh -L 7104:xuchang-lab2.staff.sydney.edu.au:7004 ivan@xuchang-lab2.staff.sydney.edu.au -p 8222


conda activate vllm-env
CUDA_VISIBLE_DEVICES=2 python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3-8B-Base   --max-model-len 4096 --port 7005 
ssh -L 7105:xuchang-lab2.staff.sydney.edu.au:7005 ivan@xuchang-lab2.staff.sydney.edu.au -p 8222


conda activate vllm-env
CUDA_VISIBLE_DEVICES=3 python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3-1.7B-FP8 --max-model-len 4096 --port 7000 
ssh -L 7100:xuchang-lab2.staff.sydney.edu.au:7000 ivan@xuchang-lab2.staff.sydney.edu.au -p 8222

conda activate vllm-env
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3-1.7B --max-model-len 4096 --port 7001
ssh -L 7101:xuchang-lab2.staff.sydney.edu.au:7001 ivan@xuchang-lab2.staff.sydney.edu.au -p 8222
# -----------------------------------------------------------------------------


# Lab 1 - VS Code
# -----------------------------------------------------------------------------
conda activate vllm-env
CUDA_VISIBLE_DEVICES=2 python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3-4B --max-model-len 4096 --port 5001

conda activate vllm-env
CUDA_VISIBLE_DEVICES=3 python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3-4B-Base --max-model-len 4096 --port 5002

conda activate vllm-env
CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3-1.7B-Base --max-model-len 4096 --port 5000
# -----------------------------------------------------------------------------


# Lab 3
# -----------------------------------------------------------------------------






# pending -------------------------------------------------------------------------------------
conda activate vllm-env
CUDA_VISIBLE_DEVICES=0,1 python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3-4B-FP8 --max-model-len 4096 --port 5000 --tensor-parallel-size 2



conda activate vllm-env
CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3-8B-FP8   --max-model-len 4096 --port 7003 
## CANNOT LOAD ON 24G ^^^^^^^^^^^^^^^^^^^^^^^^^^^

conda activate vllm-env
CUDA_VISIBLE_DEVICES=3 python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3-14B-Base  --max-model-len 4096 --port 7004

# 14B Models on GPU 0 --> 3
conda activate vllm-env
CUDA_VISIBLE_DEVICES=3 python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3-14B-FP8   --max-model-len 4096 --port 7003


## HUANTAO 
python -m LLM-Calibration-Study.simple_evals --model  --benchmark mmlu_pro --conf_mode sampling
# or
conda activate vllm
CUDA_VISIBLE_DEVICES=7 python -m vllm.entrypoints.openai.api_server --model $MODEL_DIR --max-model-len 4096 --port 8000 