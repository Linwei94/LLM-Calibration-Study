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
    "Qwen/Qwen3-235B-A22B-fp8-tput-think"
    "Qwen/Qwen3-235B-A22B-fp8-tput"
    "gemini-2.5-pro-exp-03-25"
    "gemini-2.0-flash"
    "gemini-2.0-flash-lite"
    "gemini-1.5-flash"
    "gemini-1.5-flash-8b"
    "gemini-1.5-pro"
    "grok-3-beta"
    "grok-3-mini-beta"
    "grok-2-1212"
    "claude-3-7-sonnet-20250219"
    "claude-3-5-haiku-20241022"
    "claude-3-haiku-20240307"
)

for model in "${models[@]}"; do
    echo $model
    python -m LLM-Calibration-Study.simple_evals --model $model --benchmark mmlu_pro --conf_mode eval_all &
done
