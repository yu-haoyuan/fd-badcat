#!/bin/bash
set -e

SCRIPT_DIR=$(cd $(dirname "$0") && pwd)
ROOT_DIR=$(dirname "$SCRIPT_DIR")
ASR_DIR="$ROOT_DIR/model/sherpa-onnx-paraformer-zh-2024-03-09"

ASR_URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-paraformer-zh-2024-03-09.tar.bz2"
ASR_TAR="$ROOT_DIR/model/sherpa-onnx-paraformer-zh-2024-03-09.tar.bz2"

echo "[1/3] 下载并解压 ASR 模型"
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

echo "[2/3] 检查模型状态"

if [ -d "$ASR_DIR" ]; then
    echo "ASR 模型存在"
else
    echo "ASR 模型缺失"
    exit 1
fi

check_port() {
    local PORT_HEX=$(printf "%04X" "$1")
    if grep -q ":$PORT_HEX" /proc/net/tcp; then
        echo "端口 $1 正常"
    else
        echo "端口 $1 未启动"
    fi
}

echo "[3/3] 检查服务端口"
check_port 19000
check_port 10004

echo "检测完成"
