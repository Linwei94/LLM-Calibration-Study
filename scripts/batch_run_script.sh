cd  scripts




N_samples=1000 # 0 to run all examples


bash run.sh google-llama-3.1-8b-instruct-maas $N_samples verbal mmlu
bash run.sh google-llama-3.1-70b-instruct-maas $N_samples verbal mmlu
bash run.sh google-llama-3.1-405b-instruct-maas $N_samples verbal mmlu

bash run.sh google-llama-3.1-8b-instruct-maas $N_samples verbal_cot mmlu
bash run.sh google-llama-3.1-70b-instruct-maas $N_samples verbal_cot mmlu
bash run.sh google-llama-3.1-405b-instruct-maas $N_samples verbal_cot mmlu

bash run.sh gpt-4o-mini $N_samples verbal mmlu
bash run.sh gpt-4o-mini $N_samples verbal_cot mmlu