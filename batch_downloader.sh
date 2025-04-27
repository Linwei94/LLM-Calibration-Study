#!/bin/bash

# 定义要处理的模型列表（你可以自由添加）
MODEL_LIST=(
    google/gemma-3-27b-it
    meta-llama/Llama-3.1-70B-Instruct
    meta-llama/Llama-2-70B-Instruct
    Qwen/Qwen2.5-7B-Instruct
    Qwen/Qwen2.5-14B-Instruct
    Qwen/Qwen2.5-32B-Instruct
    Qwen/Qwen2.5-32B-Instruct-GGUF
    Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4
    Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8
    Qwen/Qwen2.5-72B-Instruct
    Qwen/QwQ-32B-AWQ
    Qwen/QwQ-32B-GGUF
    Qwen/QwQ-32B
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
