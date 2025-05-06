#!/bin/bash

# 定义要处理的模型列表（你可以自由添加）
MODEL_LIST=(
    Qwen/Qwen3-0.6B
    Qwen/Qwen3-0.6B-FP8
    Qwen/Qwen3-0.6B-Base
    Qwen/Qwen3-1.7B-FP8
    Qwen/Qwen3-1.7B
    Qwen/Qwen3-1.7B-Base
    Qwen/Qwen3-4B-FP8
    Qwen/Qwen3-4B
    Qwen/Qwen3-4B-Base
    Qwen/Qwen3-8B-FP8
    Qwen/Qwen3-8B
    Qwen/Qwen3-8B-Base
    Qwen/Qwen3-14B-FP8
    Qwen/Qwen3-14B-Base
    Qwen/Qwen3-14B
    Qwen/Qwen3-30B-A3B-Base
    Qwen/Qwen3-30B-A3B-FP8
    Qwen/Qwen3-30B-A3B
    Qwen/Qwen3-32B-FP8
    Qwen/Qwen3-32B
    google/gemma-3-27b-it
    google/gemma-2-27b-it
    google/gemma-2-9b-it
    google/gemma-2b-it
    google/gemma-3-1b-it
    google/gemma-3-4b-it
    google/gemma-3-12b-it
    mistralai/Mixtral-8x7B-Instruct-v0.1
    mistralai/Mixtral-8x7B-v0.1
    mistralai/Mistral-7B-Instruct-v0.1
    mistralai/Mistral-Small-3.1-24B-Base-2503
    mistralai/Mistral-Large-Instruct-2411
    mistralai/Mistral-Nemo-Instruct-2407
    deepseek-ai/DeepSeek-R1-Distill-Llama-70B
    deepseek-ai/DeepSeek-R1-Distill-Llama-8B
    deepseek-ai/DeepSeek-R1-Distill-Qwen-32B
)

# 遍历每个模型
for MODEL_NAME in "${MODEL_LIST[@]}"; do
  echo "================================================================="
  echo "🚀 开始处理模型: $MODEL_NAME"
  echo "================================================================="

  # 调用单个模型处理脚本
  bash upload_model.sh "$MODEL_NAME" 

  # 检查上一步是否成功
  if [ $? -ne 0 ]; then
    echo "❌ 处理失败: $MODEL_NAME"
    echo "❗ 中断执行。"
    exit 1
  fi

  echo "✅ 模型 $MODEL_NAME 处理完成！"
  echo
done

echo "🏁 所有模型处理完毕！"
