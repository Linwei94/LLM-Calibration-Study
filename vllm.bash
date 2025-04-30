#!/bin/bash


# Lab 1
# models=(
#     Qwen/Qwen3-32B-FP8
#     Qwen/Qwen3-32B
#     Qwen/Qwen3-30B-A3B-Base
#     Qwen/Qwen3-30B-A3B-FP8
#     Qwen/Qwen3-4B-FP8
#     Qwen/Qwen3-4B
#     Qwen/Qwen3-4B-Base
# )


# Lab 0
# models=(
#     Qwen/Qwen3-0.6B-Base
#     Qwen/Qwen3-0.6B-FP8
#     Qwen/Qwen3-0.6B
#     Qwen/Qwen3-14B-FP8 
#     Qwen/Qwen3-14B-Base
#     Qwen/Qwen3-14B
# )


# Lab 2
models=(
    Qwen/Qwen3-1.7B-FP8
    Qwen/Qwen3-1.7B
    Qwen/Qwen3-1.7B-Base
    Qwen/Qwen3-8B-FP8
    Qwen/Qwen3-8B
    Qwen/Qwen3-8B-Base
)


for model in "${models[@]}"; do
  huggingface-cli download "$model" &
done

wait  # Wait for all background jobs to finish