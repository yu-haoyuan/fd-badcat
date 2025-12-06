FROM nvidia/cuda:12.4.1-devel-ubuntu22.04
RUN sed -i 's/archive.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list && \
    sed -i 's/security.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app


# =========================
# 系统基础依赖
# =========================
RUN apt-get update && apt-get install -y \
    git wget curl tmux ffmpeg build-essential \
    ca-certificates netcat \
    && rm -rf /var/lib/apt/lists/*

# =========================
# 安装 Miniconda
# =========================
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh

ENV PATH=/opt/conda/bin:$PATH
RUN conda init bash

# =========================
# 创建三个 Conda 环境
# =========================
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

RUN conda create -n vllm python=3.10 -y
RUN conda create -n index-tts-vllm python=3.10 -y
RUN conda create -n fd-sds python=3.10 -y

# =========================
# 安装 vLLM + Qwen3 依赖
# =========================
RUN conda run -n vllm pip install --upgrade pip
RUN mkdir -p /app/model && cd /app/model && \
    git clone -b qwen3_omni https://github.com/wangxiongts/vllm.git

WORKDIR /app/model/vllm

ENV VLLM_PRECOMPILED_WHEEL_LOCATION=
ENV VLLM_USE_PRECOMPILED=0
RUN conda run -n vllm pip install -vvv -r requirements/build.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN conda run -n vllm pip install -vvv -r requirements/cuda.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN conda run -n vllm pip install . -v --no-build-isolation

RUN conda run -n vllm pip install \
    accelerate \
    qwen-omni-utils \
    git+https://ghfast.top/https://github.com/huggingface/transformers \
    flash-attn --no-build-isolation



# =========================
# 安装 Index-TTS 环境
# =========================
WORKDIR /app/model
RUN git clone https://github.com/Ksuriuri/index-tts-vllm.git

WORKDIR /app/model/index-tts-vllm
RUN conda run -n index-tts-vllm pip install --upgrade pip
RUN conda run -n index-tts-vllm pip install -r requirements.txt

# =========================
# 安装 Agent(fd-sds) 环境
# =========================
WORKDIR /app
COPY requirements.txt /app/requirements.txt

RUN conda run -n fd-sds pip install --upgrade pip
RUN conda run -n fd-sds pip install -r /app/requirements.txt

# =========================
# 默认入口（后续你会换成 run.sh）
# =========================
CMD ["bash"]
