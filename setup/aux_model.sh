#!/bin/bash
set -e

ASR_URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-paraformer-zh-2024-03-09.tar.bz2"
ASR_TAR="sherpa-onnx-paraformer-zh-2024-03-09.tar.bz2"
ASR_DIR="../model/sherpa-onnx-paraformer-zh-2024-03-09"

echo "[1/3] 下载并解压 ASR 模型"
mkdir -p ../model
cd ../model

if [ ! -d "sherpa-onnx-paraformer-zh-2024-03-09" ]; then
    wget -q $ASR_URL -O $ASR_TAR
    tar xf $ASR_TAR
    rm $ASR_TAR
else
    echo "ASR 模型已存在，跳过下载"
fi

cd ../setup

echo "[2/3] 检查模型状态"
if [ -d "../model/sherpa-onnx-paraformer-zh-2024-03-09" ]; then
    echo "✔ ASR 模型存在"
else
    echo "✘ ASR 模型缺失"
    exit 1
fi

echo "[3/3] 检查服务端口"

check_port() {
    local PORT=$1
    nc -z 127.0.0.1 $PORT >/dev/null 2>&1
    if [ $? -eq 0 ]; then
        echo "✔ 端口 $PORT 正常"
    else
        echo "✘ 端口 $PORT 未启动"
    fi
}

check_port 19000
check_port 10004

echo "检测完成"
