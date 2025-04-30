#!/bin/bash

models=(
    Qwen/Qwen3-32B-FP8
    Qwen/Qwen3-32B
    Qwen/Qwen3-30B-A3B-Base
    Qwen/Qwen3-30B-A3B-FP8
    Qwen/Qwen3-1.7B-FP8
    Qwen/Qwen3-1.7B
    Qwen/Qwen3-1.7B-Base
)



for model in "${models[@]}"; do
  huggingface-cli download "$model" &
done

wait  # Wait for all background jobs to finish