# Free Together AI

# MODEL="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free" # with logprobs
MODEL="gpt-4.1-nano" # with logprobs
# MODEL="meta/llama-4-maverick-17b-128e-instruct-maas" # no logprobs

BENCHMARK="mmlu_pro"
# CONFS=("sampling" "verbal_numerical_shared_sampling"  "logit_perplexity_shared_sampling" "verbal_linguistic_shared_sampling" "token_sar_shared_sampling")
CONFS=("verbal_numerical_shared_sampling"  "logit_perplexity_shared_sampling" "verbal_linguistic_shared_sampling" "token_sar_shared_sampling")
N_EXAMPLES=3

# for CONF in "${CONFS[@]}"; do
#     if [[ "$CONF" != "sampling" ]]; then
#         (python -m LLM-Calibration-Study.simple_evals --model "$MODEL" --benchmark "$BENCHMARK" --conf_mode "$CONF" --examples "$N_EXAMPLES" &)
#     else
#         (python -m LLM-Calibration-Study.simple_evals --model "$MODEL" --benchmark "$BENCHMARK" --conf_mode "$CONF" --examples "$N_EXAMPLES")
#     fi
# done

for CONF in "${CONFS[@]}"; do
    if [[ "$CONF" != "sampling" ]]; then
        (python -m LLM-Calibration-Study.simple_evals --model "$MODEL" --benchmark "$BENCHMARK" --conf_mode "$CONF" &)
    else
        (python -m LLM-Calibration-Study.simple_evals --model "$MODEL" --benchmark "$BENCHMARK" --conf_mode "$CONF")
    fi
done