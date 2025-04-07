N_samples=100 # 0 to run all examples

# bash scripts/run.sh databricks-llama-4-maverick $N_samples verbal gpqa --debug
# bash scripts/run.sh databricks-llama-4-maverick $N_samples verbal_cot gpqa --debug


bash scripts/run.sh Llama-4-Scout-17B-16E-Instruct $N_samples verbal gpqa --debug
bash scripts/run.sh Llama-4-Scout-17B-16E-Instruct $N_samples verbal_cot gpqa --debug
bash scripts/run.sh Llama-4-Scout-17B-16E-Instruct 1000 verbal mmlu
bash scripts/run.sh Llama-4-Scout-17B-16E-Instruct 1000 verbal_cot mmlu


# bash scripts/run.sh google-llama-3.1-8b-instruct-maas $N_samples verbal gpqa --debug
# bash scripts/run.sh google-llama-3.1-8b-instruct-maas $N_samples verbal_cot gpqa --debug

# bash scripts/run.sh google-llama-3.1-70b-instruct-maas $N_samples verbal gpqa --debug
# bash scripts/run.sh google-llama-3.1-70b-instruct-maas $N_samples verbal_cot gpqa --debug

# bash scripts/run.sh google-llama-3.1-405b-instruct-maas $N_samples verbal gpqa
# bash scripts/run.sh google-llama-3.1-405b-instruct-maas $N_samples verbal_cot gpqa

# bash scripts/run.sh gpt-4o-mini $N_samples verbal gpqa --debug
# bash scripts/run.sh gpt-4o-mini $N_samples verbal_cot gpqa --debug

# bash scripts/run.sh gpt-4o-2024-11-20_assistant $N_samples verbal gpqa --debug
# bash scripts/run.sh gpt-4o-2024-11-20_assistant $N_samples verbal_cot gpqa --debug