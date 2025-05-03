#!/bin/bash

# 检查参数
if [ $# -lt 1 ]; then
  echo "Usage: $0 <model_name> [optional: remote_server_user@ip]"
  exit 1
fi


MODEL_NAME=$1
REMOTE=${2:-huangtao3@101.230.144.229}  # 默认远程服务器账号@IP
REMOTE_PORT=8003
LOCAL_DIR="/hdd/.cache/huggingface"
REMOTE_DIR="~/.cache/huggingface/hub/"

# 把 / 替换成 --
MODEL_CACHE_DIR="models--${MODEL_NAME//\//--}"

# 下载模型到本地
echo "🔵 Downloading model: $MODEL_NAME ..."
huggingface-cli download "$MODEL_NAME"

# 检查是否成功
if [ ! -d $LOCAL_DIR/hub/$MODEL_CACHE_DIR ]; then
  echo "❌ Model download failed!"
  exit 1
fi

# 上传模型到远程服务器
echo "🔵 Uploading model to server: $REMOTE ..."
TMPDIR=~/git_tmp rsync -avzP -e "ssh -p $REMOTE_PORT" $LOCAL_DIR/hub/$MODEL_CACHE_DIR "$REMOTE:$REMOTE_DIR"

# 检查是否上传成功
if [ $? -ne 0 ]; then
  echo "❌ SCP failed!"
  exit 1
fi

# 删除本地模型
echo "🧹 Cleaning up local model cache..."
rm -rf $LOCAL_DIR/hub/$MODEL_CACHE_DIR

echo "✅ Done!"
