# Free Together AI

# MODEL="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free" # with logprobs
# MODEL="gpt-4.1-mini" # with logprobs
MODEL=meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo # with logprobs
# MODEL="meta/llama-4-maverick-17b-128e-instruct-maas" # no logprobs

BENCHMARK="mmlu_pro"
# CONFS=("sampling" "verbal_numerical_shared_sampling"  "logit_perplexity_shared_sampling" "verbal_linguistic_shared_sampling" "token_sar_shared_sampling")
CONFS=("verbal_numerical_shared_sampling"  "logit_perplexity_shared_sampling" "verbal_linguistic_shared_sampling" "token_sar_shared_sampling")
CONFS=("token_sar_shared_sampling")
N_EXAMPLES=1000

for CONF in "${CONFS[@]}"; do
    if [[ "$CONF" != "sampling" ]]; then
        (python -m LLM-Calibration-Study.simple_evals --model "$MODEL" --benchmark "$BENCHMARK" --conf_mode "$CONF" --examples "$N_EXAMPLES" &)
    else
        (python -m LLM-Calibration-Study.simple_evals --model "$MODEL" --benchmark "$BENCHMARK" --conf_mode "$CONF" --examples "$N_EXAMPLES")
    fi
done

# for CONF in "${CONFS[@]}"; do
#     if [[ "$CONF" != "sampling" ]]; then
#         (python -m LLM-Calibration-Study.simple_evals --model "$MODEL" --benchmark "$BENCHMARK" --conf_mode "$CONF" &)
#     else
#         (python -m LLM-Calibration-Study.simple_evals --model "$MODEL" --benchmark "$BENCHMARK" --conf_mode "$CONF")
#     fi
# done



# done: 
python -m LLM-Calibration-Study.simple_evals --model meta-llama/Llama-3.2-3B-Instruct-Turbo --benchmark mmlu_pro --conf_mode verbal_linguistic_shared_sampling 
# not working"
# python -m LLM-Calibration-Study.simple_evals --model meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8 --benchmark mmlu_pro --conf_mode sampling
# python -m LLM-Calibration-Study.simple_evals --model meta-llama/Llama-4-Scout-17B-16E-Instruct --benchmark mmlu_pro --conf_mode sampling

#Pending:
python -m LLM-Calibration-Study.simple_evals --model meta-llama/Llama-3.3-70B-Instruct-Turbo --benchmark mmlu_pro --conf_mode sampling


# Currently running:
python -m LLM-Calibration-Study.simple_evals --model meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo --benchmark mmlu_pro --conf_mode sampling
python -m LLM-Calibration-Study.simple_evals --model meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo --benchmark mmlu_pro --conf_mode sampling
python -m LLM-Calibration-Study.simple_evals --model meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo --benchmark mmlu_pro --conf_mode sampling
python -m LLM-Calibration-Study.simple_evals --model meta-llama/Llama-2-70b-hf --benchmark mmlu_pro --conf_mode sampling

# 0.06 usd


python -m LLM-Calibration-Study.simple_evals --model meta-llama/Llama-3.2-3B-Instruct-Turbo --benchmark mmlu_pro --conf_mode verbal_linguistic_shared_sampling 
python -m LLM-Calibration-Study.simple_evals --model meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo --benchmark mmlu_pro --conf_mode verbal_linguistic_shared_sampling
python -m LLM-Calibration-Study.simple_evals --model meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo --benchmark mmlu_pro --conf_mode verbal_linguistic_shared_sampling


python -m LLM-Calibration-Study.simple_evals --model meta-llama/Llama-2-70b-hf --benchmark mmlu_pro --conf_mode verbal_numerical_shared_sampling
python -m LLM-Calibration-Study.simple_evals --model meta-llama/Llama-2-70b-hf --benchmark mmlu_pro --conf_mode logit_perplexity_shared_sampling


python -m LLM-Calibration-Study.simple_evals --model deepseek-ai/DeepSeek-R1 --benchmark mmlu_pro --conf_mode sampling --examples 3

conda activate llm-uncertainty
python -m LLM-Calibration-Study.simple_evals --model gemini-2.5-flash-preview-04-17 --benchmark mmlu_pro --conf_mode sampling

conda activate llm-uncertainty
python -m LLM-Calibration-Study.simple_evals --model gemini-2.5-pro-exp-03-25 --benchmark mmlu_pro --conf_mode sampling

conda activate llm-uncertainty
python -m LLM-Calibration-Study.simple_evals --model gemini-2.0-flash --benchmark mmlu_pro --conf_mode sampling

conda activate llm-uncertainty
python -m LLM-Calibration-Study.simple_evals --model gemini-2.0-flash-lite --benchmark mmlu_pro --conf_mode sampling

conda activate llm-uncertainty
python -m LLM-Calibration-Study.simple_evals --model gemini-1.5-flash --benchmark mmlu_pro --conf_mode sampling

conda activate llm-uncertainty
python -m LLM-Calibration-Study.simple_evals --model gemini-1.5-flash-8b --benchmark mmlu_pro --conf_mode sampling

conda activate llm-uncertainty
python -m LLM-Calibration-Study.simple_evals --model gemini-1.5-pro --benchmark mmlu_pro --conf_mode sampling





"Qwen/Qwen3-8B-Base",
"Qwen/Qwen3-14B-FP8",
"Qwen/Qwen3-14B-Base",
"Qwen/Qwen3-32B-FP8",
"Qwen/Qwen3-32B",
"Qwen/Qwen3-30B-A3B-Base",
"Qwen/Qwen3-30B-A3B-FP8"

# Lab 1
# CUDA_VISIBLE_DEVICES=0 python -m LLM-Calibration-Study.simple_evals --model Qwen/Qwen3-0.6B-Base --benchmark mmlu_pro --conf_mode sampling
CUDA_VISIBLE_DEVICES=0 python -m LLM-Calibration-Study.simple_evals --model Qwen/Qwen3-8B-Base --benchmark mmlu_pro --conf_mode sampling
CUDA_VISIBLE_DEVICES=1 python -m LLM-Calibration-Study.simple_evals --model Qwen/Qwen3-0.6B --benchmark mmlu_pro --conf_mode sampling
CUDA_VISIBLE_DEVICES=3 python -m LLM-Calibration-Study.simple_evals --model Qwen/Qwen3-0.6B-FP8 --benchmark mmlu_pro --conf_mode sampling
CUDA_VISIBLE_DEVICES=2 python -m LLM-Calibration-Study.simple_evals --model Qwen/Qwen3-4B --benchmark mmlu_pro --conf_mode sampling


# Lab 2
CUDA_VISIBLE_DEVICES=0 python -m LLM-Calibration-Study.simple_evals --model Qwen/Qwen3-1.7B-FP8 --benchmark mmlu_pro --conf_mode sampling
CUDA_VISIBLE_DEVICES=1 python -m LLM-Calibration-Study.simple_evals --model Qwen/Qwen3-1.7B --benchmark mmlu_pro --conf_mode sampling
CUDA_VISIBLE_DEVICES=3 python -m LLM-Calibration-Study.simple_evals --model Qwen/Qwen3-1.7B-Base --benchmark mmlu_pro --conf_mode sampling
CUDA_VISIBLE_DEVICES=2 python -m LLM-Calibration-Study.simple_evals --model Qwen/Qwen3-4B-FP8 --benchmark mmlu_pro --conf_mode sampling

# Lab 0: device 3 is actually 0
CUDA_VISIBLE_DEVICES=3 python -m LLM-Calibration-Study.simple_evals --model Qwen/Qwen3-14B --benchmark mmlu_pro --conf_mode sampling
CUDA_VISIBLE_DEVICES=0 python -m LLM-Calibration-Study.simple_evals --model Qwen/Qwen3-8B-FP8 --benchmark mmlu_pro --conf_mode sampling
CUDA_VISIBLE_DEVICES=1 python -m LLM-Calibration-Study.simple_evals --model Qwen/Qwen3-4B-Base --benchmark mmlu_pro --conf_mode sampling
CUDA_VISIBLE_DEVICES=2 python -m LLM-Calibration-Study.simple_evals --model Qwen/Qwen3-8B --benchmark mmlu_pro --conf_mode sampling


# Overseas
CUDA_VISIBLE_DEVICES=1,2,3 python -m LLM-Calibration-Study.simple_evals --model Qwen/Qwen3-32B --benchmark mmlu_pro --conf_mode sampling
