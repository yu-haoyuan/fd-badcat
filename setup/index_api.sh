#!/bin/bash
set -e

echo "[1/4] 下载 Index-TTS-1.5-vLLM 模型..."
mkdir -p ../model
modelscope download \
    --model kusuriuri/Index-TTS-1.5-vLLM \
    --local_dir ../model/Index-TTS-1.5-vLLM

echo "[2/4] 下载 index-tts-vllm 仓库..."
if [ ! -d "../model/index-tts-vllm" ]; then
    git clone https://github.com/Ksuriuri/index-tts-vllm.git ../model/index-tts-vllm
else
    echo "仓库已存在，跳过。"
fi

echo "[3/4] 创建 conda 环境 index-tts-vllm ..."
conda create -n index-tts-vllm python=3.10 -y
source ~/.bashrc
conda activate index-tts-vllm

echo "[4/4] 在 tmux 中启动 TTS 服务..."
tmux kill-session -t index-tts 2>/dev/null || true

tmux new-session -d -s index-tts \
    "cd ../model/index-tts-vllm && \
     source ~/.bashrc && conda activate index-tts-vllm && \
     pip install -r requirements.txt && \
     pip install vllm-cu12 || pip install vllm && \
     python model/index-tts-vllm/api_example.py \
        --host 0.0.0.0 \
        --port 19000 \
        --model_dir ../../model/Index-TTS-1.5-vLLM \
        --gpu_memory_utilization 0.8"

echo "启动完成。查看： tmux attach -t index-tts"
