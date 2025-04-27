# Example: How to deploy Qwen-2.5-32B-Instruct-AWQ with vLLM on Taohuang's server
## Step 1: Download at your local machine
```bash
huggingface-cli download Qwen/Qwen2.5-32B-Instruct-AWQ
```

## Step 2: Since 101.230.144.229 cannot download from huggingface, we scp the model to Tao Huang's server
```bash
scp -P 8003 -r ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-32B-Instruct-AWQ huangtao3@101.230.144.229:~/.cache/huggingface/hub/
```



## Step 2: Start a vLLM service on Tao Huang's server
```bash
screen -S vllm
```
```bash
export MODEL_ID="Qwen/Qwen2.5-32B-Instruct-AWQ"
export MODEL_DIR=$(ls -d ~/.cache/huggingface/hub/models--${MODEL_ID//\//--}/snapshots/* | head -n 1)

CUDA_VISIBLE_DEVICES=7 \
python3 -m vllm.entrypoints.openai.api_server \
  --model $MODEL_DIR \
  --served-model-name "$MODEL_ID" \
  --quantization awq \
  --host 0.0.0.0 \
  --port 8000
```

## Step 4: Start a SSH tunnel on your local machine
```bash
ssh -p 8003 -L 8000:localhost:8000 huangtao3@101.230.144.229
```


## Step 5: Call the vLLM service on your local machine
```bash
curl http://localhost:8000/v1/completions -H "Content-Type: application/json" -d '{"model": "Qwen/Qwen2.5-32B-Instruct-AWQ", "prompt": "Hello, Qwen.", "max_tokens": 100}'
```



