#!/bin/bash
set -e

ENV_NAME="fd-sds"

echo "[1/3] 创建 conda 环境 (如不存在)"
source ~/.bashrc
conda env list | grep -q "$ENV_NAME" || conda create -n $ENV_NAME python=3.10 -y

echo "[2/3] 安装 Python 依赖"
conda activate $ENV_NAME
pip install -r requirements.txt

echo "[3/3] 启动服务 (backend + frontend)"
tmux kill-session -t fd 2>/dev/null || true

tmux new-session -d -s fd -n backend "bash -lc '
    source ~/.bashrc;
    conda activate $ENV_NAME;
    python src/backend.py --config src/config.yaml
'"

sleep 15

tmux new-window -t fd:1 -n frontend "bash -lc '
    source ~/.bashrc;
    conda activate $ENV_NAME;
    python src/frontend.py --config src/config.yaml
'"

echo "启动完成: tmux attach -t fd"
tmux attach -t fd
