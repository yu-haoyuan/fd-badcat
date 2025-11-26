#!/bin/bash
set -e
# 永远正确加载 conda（无论用户是否初始化）
if [ -f "/root/miniconda3/etc/profile.d/conda.sh" ]; then
    source /root/miniconda3/etc/profile.d/conda.sh
else
    export PATH="/root/miniconda3/bin:$PATH"
fi
SESSION="qwen3"
ENV_NAME="vllm"
MODEL_DIR="./model/qwen3omni"
PORT_VLLM=10003
PORT_PROXY=10004

echo "=== [1/4] Conda 环境准备（统一放在一起） ==="

# 关键修复：使用 conda 官方推荐的、永不报错的激活方式
eval "$(conda shell.bash hook)"

INSTALL_DEPS=0
if conda env list | grep -q "$ENV_NAME"; then
    echo "环境 $ENV_NAME 已存在，跳过创建与依赖安装"
else
    echo "环境不存在，开始创建..."
    conda create -y -n $ENV_NAME python=3.10
    INSTALL_DEPS=1
fi

echo "激活环境：$ENV_NAME"
conda activate $ENV_NAME


if [ $INSTALL_DEPS -eq 1 ]; then
    echo "首次创建环境 → 安装所有依赖（vLLM / flash-attn / transformers）"

    mkdir -p model
    cd model

    if [ ! -d "vllm" ]; then
        git clone -b qwen3_omni https://github.com/wangxiongts/vllm.git
    fi

    cd vllm
    pip install -r requirements/build.txt
    pip install -r requirements/cuda.txt

    export VLLM_PRECOMPILED_WHEEL_LOCATION="https://wheels.vllm.ai/a5dd03c1ebc5e4f56f3c9d3dc0436e9c582c978f/vllm-0.9.2-cp38-abi3-manylinux1_x86_64.whl"
    VLLM_USE_PRECOMPILED=1 pip install -e . -v --no-build-isolation

    pip install accelerate
    pip install -U flash-attn --no-build-isolation
    pip install qwen-omni-utils -U
    pip install git+https://github.com/huggingface/transformers

    cd ../..
else
    echo "环境已存在 → 跳过依赖安装"
fi


echo "=== [2/4] 下载 Qwen3-Omni 模型 ==="
mkdir -p $MODEL_DIR
SAFETENSOR_COUNT=$(find "$MODEL_DIR" -maxdepth 1 -name "*.safetensors" | wc -l)

if [ "$SAFETENSOR_COUNT" -ge 15 ]; then
    echo "检测到 $SAFETENSOR_COUNT 个 safetensors 文件，模型已存在，跳过下载"
else
    echo "当前 safetensors 数量：$SAFETENSOR_COUNT (< 15)，开始下载模型..."
    modelscope download \
        --model Qwen/Qwen3-Omni-30B-A3B-Instruct \
        --local_dir $MODEL_DIR
fi


echo "=== [3/4] 启动 vLLM (tmux) ==="
tmux kill-session -t $SESSION 2>/dev/null || true

# 第一个窗口：vLLM
tmux new-session -d -s $SESSION -n vllm "bash -lc '
    eval \"\$(conda shell.bash hook)\";
    conda activate $ENV_NAME;
    vllm serve $MODEL_DIR \
        --port $PORT_VLLM \
        --host 0.0.0.0 \
        --dtype bfloat16 \
        --max-model-len 32768 \
        --allowed-local-media-path / \
        -tp 4
'"

# 第二个窗口：API
tmux new-window -t $SESSION:1 -n api "bash -lc '
    eval \"\$(conda shell.bash hook)\";
    conda activate $ENV_NAME;
    python src/qwen3_api.py --port $PORT_PROXY
'"

# 启动时自动进入 vLLM 窗口
tmux select-window -t $SESSION:1
tmux attach -t $SESSION


# vllm serve /root/autodl-tmp/fd-badcat/model/qwen3omni \
#     --port 10003 \
#     --host 0.0.0.0 \
#     --dtype bfloat16 \
#     --max-model-len 32768 \
#     --allowed-local-media-path / \
#     -tp 4