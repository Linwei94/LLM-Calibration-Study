#!/bin/bash

models=(
    "deepseek-chat"
    "deepseek-reasoner"
    "gpt-4.1-mini"
    "gpt-4.1-nano"
    "meta-llama/Llama-3.2-3B-Instruct-Turbo"
    "meta-llama/Llama-4-Scout-17B-16E-Instruct"
    "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"
    "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
    "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"
    "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo"
    "Qwen/Qwen3-235B-A22B-fp8"
)

for model in "${models[@]}"; do
     python -m LLM-Calibration-Study.simple_evals --model $model --benchmark mmlu_pro --conf_mode eval_all &
done
