# Free Together AI
# python -m LLM-Calibration-Study.simple_evals --model meta-llama/Llama-3.3-70B-Instruct-Turbo-Free --benchmark gpqa --conf_mode sampling
# python -m LLM-Calibration-Study.simple_evals --model meta-llama/Llama-3.3-70B-Instruct-Turbo-Free --benchmark gpqa --conf_mode verbal_numerical_shared_sampling 
# python -m LLM-Calibration-Study.simple_evals --model meta-llama/Llama-3.3-70B-Instruct-Turbo-Free --benchmark gpqa --conf_mode logit_perplexity_shared_sampling 
# python -m LLM-Calibration-Study.simple_evals --model meta-llama/Llama-3.3-70B-Instruct-Turbo-Free --benchmark gpqa --conf_mode verbal_linguistic_shared_sampling 

python -m LLM-Calibration-Study.simple_evals --model meta-llama/Llama-3.3-70B-Instruct-Turbo-Free --benchmark simpleqa --conf_mode sampling
python -m LLM-Calibration-Study.simple_evals --model meta-llama/Llama-3.3-70B-Instruct-Turbo-Free --benchmark simpleqa --conf_mode verbal_numerical_shared_sampling 
python -m LLM-Calibration-Study.simple_evals --model meta-llama/Llama-3.3-70B-Instruct-Turbo-Free --benchmark simpleqa --conf_mode logit_perplexity_shared_sampling 
python -m LLM-Calibration-Study.simple_evals --model meta-llama/Llama-3.3-70B-Instruct-Turbo-Free --benchmark simpleqa --conf_mode verbal_linguistic_shared_sampling 
