#!/usr/bin/env bash
mkdir model
cd model
git clone -b qwen3_omni https://github.com/wangxiongts/vllm.git
conda create -n fdbc-qwen3o-vllm python=3.10
conda activate fdbc-qwen3o-vllm
cd vllm
pip install -r requirements/build.txt
pip install -r requirements/cuda.txt
pip install -e . -v --no-build-isolation
pip install tokenizers==0.22.1 transformers==4.57.3
pip install accelerate
pip install qwen-omni-utils -U
pip install -U flash-attn --no-build-isolation
modelscope download --model Qwen/Qwen3-Omni-30B-A3B-Instruct --local_dir ./model/Qwen3-Omni-30B-A3B-Instruct
