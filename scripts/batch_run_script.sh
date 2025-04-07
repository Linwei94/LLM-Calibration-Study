N_samples=10 # 0 to run all examples

bash scripts/run.sh databricks-llama-4-maverick $N_samples verbal mmlu --debug


# bash scripts/run.sh google-llama-3.1-8b-instruct-maas $N_samples verbal gpqa --debug
# bash scripts/run.sh google-llama-3.1-8b-instruct-maas $N_samples verbal_cot gpqa --debug

# bash scripts/run.sh google-llama-3.1-70b-instruct-maas $N_samples verbal gpqa --debug
# bash scripts/run.sh google-llama-3.1-70b-instruct-maas $N_samples verbal_cot gpqa --debug

# bash scripts/run.sh google-llama-3.1-405b-instruct-maas $N_samples verbal gpqa
# bash scripts/run.sh google-llama-3.1-405b-instruct-maas $N_samples verbal_cot gpqa

# bash scripts/run.sh gpt-4o-mini $N_samples verbal gpqa
# bash scripts/run.sh gpt-4o-mini $N_samples verbal_cot gpqa

# bash scripts/run.sh gpt-4o-2024-11-20_assistant $N_samples verbal gpqa
# bash scripts/run.sh gpt-4o-2024-11-20_assistant $N_samples verbal_cot gpqa