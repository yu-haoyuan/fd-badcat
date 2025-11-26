#!/bin/bash
set -e

eval "$(conda shell.bash hook)"

echo "[1/4] 下载 index-tts-vllm 仓库"
if [ ! -d "./model/index-tts-vllm" ]; then
    git clone https://github.com/Ksuriuri/index-tts-vllm.git ./model/index-tts-vllm
else
    echo "仓库已存在 跳过"
fi

echo "[2/4] 创建 conda 环境 index-tts-vllm"
if conda env list | grep -q "index-tts-vllm"; then
    echo "环境已存在 跳过创建"
else
    conda create -n index-tts-vllm python=3.10 -y
fi

echo "激活环境 index-tts-vllm"
conda activate index-tts-vllm
cd ./model/index-tts-vllm
pip install -r requirements.txt
cd -

echo "[3/4] 下载 Index-TTS-1.5-vLLM 模型"

MODEL_PATH="./model/Index-TTS-1.5-vLLM"

mkdir -p ./model
pip install -U modelscope

if [ -d "$MODEL_PATH" ] && [ "$(ls -A $MODEL_PATH)" ]; then
    echo "模型已存在 跳过下载"
else
    echo "开始下载模型"
    modelscope download \
        --model kusuriuri/Index-TTS-1.5-vLLM \
        --local_dir "$MODEL_PATH"
fi

echo "[4/4] 在 tmux 中启动 TTS 服务"
tmux kill-session -t index-tts 2>/dev/null || true

tmux new-session -d -s index-tts \
"bash -lc 'eval \"\$(conda shell.bash hook)\" && conda activate index-tts-vllm && \
python model/index-tts-vllm/api_server.py \
  --host 0.0.0.0 \
  --port 19000 \
  --model_dir model/Index-TTS-1.5-vLLM \
  --gpu_memory_utilization 0.8'"

echo "启动完成tmux attach -t index-tts"
tmux attach -t index-tts