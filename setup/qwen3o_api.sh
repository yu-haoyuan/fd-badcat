#!/bin/bash
set -e

SESSION="qwen3"
ENV_NAME="fd-qwen"
MODEL_DIR="./model/qwen3omni"
PORT_VLLM=10003
PORT_PROXY=10004

echo "[1/5] 创建 conda 环境"
source ~/.bashrc
conda env list | grep -q "$ENV_NAME" || conda create -y -n $ENV_NAME python=3.10

echo "[2/5] 下载 Qwen3-Omni-30B 模型"
mkdir -p $MODEL_DIR
modelscope download \
    --model Qwen/Qwen3-Omni-30B-A3B-Instruct \
    --local_dir $MODEL_DIR

echo "[3/5] 安装 vLLM 及依赖"
conda activate $ENV_NAME
cd model
if [ ! -d "vllm" ]; then
    git clone -b qwen3_omni https://github.com/wangxiongts/vllm.git
fi

cd vllm
pip install -r requirements/build.txt
pip install -r requirements/cuda.txt

export VLLM_PRECOMPILED_WHEEL_LOCATION="https://wheels.vllm.ai/a5dd03c1ebc5e4f56f3c9d3dc0436e9c582c978f/vllm-0.9.2-cp38-abi3-manylinux1_x86_64.whl"
VLLM_USE_PRECOMPILED=1 pip install -e . -v --no-build-isolation

pip install git+https://github.com/huggingface/transformers
pip install accelerate
pip install qwen-omni-utils -U
pip install -U flash-attn --no-build-isolation
cd ../..

echo "[4/5] 启动 vLLM (tmux)"
tmux kill-session -t $SESSION 2>/dev/null || true

tmux new-session -d -s $SESSION -n vllm "bash -lc '
    source ~/.bashrc;
    conda activate $ENV_NAME;
    vllm serve $MODEL_DIR \
        --port $PORT_VLLM \
        --host 0.0.0.0 \
        --dtype bfloat16 \
        --max-model-len 32768 \
        --allowed-local-media-path / \
        -tp 4
'"

sleep 180

echo "[5/5] 启动 Proxy API (tmux)"
tmux new-window -t $SESSION:1 -n api "bash -lc '
    source ~/.bashrc;
    conda activate $ENV_NAME;
    python qwen3_api.py --port $PORT_PROXY
'"

echo "启动完成： tmux attach -t $SESSION"
