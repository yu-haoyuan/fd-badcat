#!/bin/bash
set -e

eval "$(conda shell.bash hook)"
source ~/.bashrc
conda activate fd-sds

pip install -r requirements.txt

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
