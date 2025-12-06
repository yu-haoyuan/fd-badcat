#!/bin/bash
set -e

SCRIPT_DIR=$(cd $(dirname "$0") && pwd)
ROOT_DIR=$(dirname "$SCRIPT_DIR")
ASR_DIR="$ROOT_DIR/model/sherpa-onnx-paraformer-zh-2024-03-09"

ASR_URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-paraformer-zh-2024-03-09.tar.bz2"
ASR_TAR="$ROOT_DIR/model/sherpa-onnx-paraformer-zh-2024-03-09.tar.bz2"

eval "$(conda shell.bash hook)"
source ~/.bashrc

conda env list | grep -q fd-sds || conda create -n fd-sds python=3.10 -y
conda activate fd-sds
pip install -r "$ROOT_DIR/requirements.txt"

echo "下载并解压 ASR 模型"
mkdir -p "$ROOT_DIR/model"
cd "$ROOT_DIR/model"

if [ ! -d "sherpa-onnx-paraformer-zh-2024-03-09" ]; then
    echo "开始下载 ASR 模型"
    wget "$ASR_URL" -O "$ASR_TAR"
    echo "解压中..."
    tar xf "$ASR_TAR"
    rm "$ASR_TAR"
else
    echo "ASR 模型已存在 跳过下载"
fi

cd "$ROOT_DIR"

echo "检查模型状态"

if [ -d "$ASR_DIR" ]; then
    echo "ASR 模型存在"
else
    echo "ASR 模型缺失"
    exit 1
fi

