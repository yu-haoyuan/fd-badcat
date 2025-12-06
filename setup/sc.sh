#!/bin/bash
set -e

eval "$(conda shell.bash hook)"
source ~/.bashrc
conda activate fd-sds
check_port() {
    local PORT_HEX=$(printf "%04X" "$1")
    if grep -q ":$PORT_HEX" /proc/net/tcp; then
        echo "端口 $1 正常"
    else
        echo "端口 $1 未启动"
    fi
}

echo "检查服务端口"
check_port 19000
check_port 10003

echo "检测完成"

tmux kill-session -t fd 2>/dev/null || true

tmux new-session -d -s fd -n backend "bash -lc '
    source ~/.bashrc;
    conda activate fd-sds;
    python src/backend.py --config src/config.yaml
'"

sleep 15

tmux new-window -t fd:1 -n frontend "bash -lc '
    source ~/.bashrc;
    conda activate fd-sds;
    python src/frontend.py --config src/config.yaml
'"

tmux attach -t fd
