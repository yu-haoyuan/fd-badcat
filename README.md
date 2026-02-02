# fd-badcat
full duplex-spoken dialogue system

> [Unit-Based Agent for Semi-Cascaded Full-Duplex Dialogue Systems](https://arxiv.org/abs/2601.20230) <br>
> [Haoyuan Yu](https://yu-haoyuan.github.io/), [Yuxuan Chen], [Minjie Cai](https://cai-mj.github.io/) <br>
> ICASSP 2026 Grand Challenge
---

Our paper is accepted by **ICASSP-2026 Grand Challenge**

![image](https://github.com/yu-haoyuan/fd-badcat/blob/main/fig.png)



---
### 环境准备

我们提供了一键式启动的docker环境：
```
docker build --progress=plain -t fd-badcat .
```
但是请注意，由于docker的国内镜像存在问题，我们在试运行时，多次在vllm编译阶段出现了不可避免的错误，所以如果`docker file`报错，请根据以下步骤手动安装环境（我们已经在本地环境确定，如下方案100%可行）：

首先，确认本机是否安装tmux：
```
command -v tmux >/dev/null 2>&1 || (sudo apt update && sudo apt install -y tmux)
```

然后在终端中，依次运行脚本：
```
bash setup/qwen3omni_env.sh
bash setup/indextts_env.sh
bash setup/aux_model.sh
```

---
### 环境检查

使用docker或是脚本安装完毕后，使用`conda env list`进行检查，正确的环境内容为：
```
fd-sds                   /root/miniconda3/envs/fd-sds(系统运行环境)
index-tts-vllm           /root/miniconda3/envs/index-tts-vllm(index服务环境)
fdbc-qwen3o-vllm         /root/miniconda3/envs/vllm(qwen3omni环境)
```

准备完毕后正确的文件子目录`model`为：
```
model/
├── Qwen3-Omni-30B-A3B-Instruct/
├── index-tts-vllm/
│   └── checkpoints/
│       └── Index-TTS-1.5-vLLM/
└── sherpa-onnx-paraformer-zh-2024-03-09/
```
---

### 数据准备

创建`exp/exp-1`文件夹，作为指定的数据目录：
```
mkdir exp/exp-1
```
然后将符合赛事要求的`test/clean`目录，放到`exp/exp-1`下面：
```
exp/
└── exp-1/
    ├── clean/
    └── test/
```
---

### 启动说明

##### 1.api启动
我们的仓库主要是基于调用`qwen3omni`的api和`indextts-1.5`的api进行实验

我们的项目逻辑比较简单，如果两个api配置无误，那么实验本身依赖的环境则不会造成困扰，只依赖最基础的前后端工具

我们的实验采取模拟现实时长的前后端模式，也就是说，原始数据的长度≈实验运行的长度，所以需要`screen`或者`tmux`进行多终端持续并发运行。

但是如果实验足够短，那么简单的多终端运行也是可以接受的。经过我们的测试，手动在多终端启动是便捷可靠的（得益于vllm的并发优化）。

**这里我们需要在五个终端全部成功启动的情况下运行实验**

请在终端1运行以下指令：
```
conda activate fdbc-qwen3o-vllm 
vllm serve model/Qwen3-Omni-30B-A3B-Instruct --port 10003 --host 0.0.0.0 --dtype bfloat16 --max-model-len 65536 --allowed-local-media-path / -tp 4
```
启动qwen3omni vllm模型，如果正确启动，会在该终端下面看到
`running on http://0.0.0.0:10003`的启动说明，请保持这个终端的开启；

请在终端2运行以下指令:
```
conda activate fdbc-qwen3o-vllm 
python src/qwen3_api.py
```
如果正确启动，会在该终端下面看到
`running on http://0.0.0.0:10004`的启动说明，请保持这个终端的开启；


请在终端3运行以下指令：
```
conda activate index-tts-vllm
python model/index-tts-vllm/api_server.py
```
启动index-tts vllm模型，如果正确启动，会在该终端下面看到
`INFO:     Uvicorn running on http://0.0.0.0:19000 (Press CTRL+C to quit)` 的启动说明，请同样保持这个终端的开启；

##### 2.主实验启动
直接运行脚本：
```
bash src/sc.sh
```
将会自动开启前后端，开始合成输出，并且提示`启动完成`，此时物理终端还是 1 个，里面有 2 个 tmux 窗口在跑服务。

如果脚本`sc.sh`一键运行失败，那么请在第4,5个终端，手动启动前后端脚本：
```
python src/backend.py --config fd-badcat/src/config.yaml
python src/frontend.py --config fd-badcat/src/config.yaml
```

最后的正确的输出为
```
exp/
└── exp-1/
    ├── clean/
    ├── HD-Track2/        ← 这是放 output 的文件夹
    │   ├── clean/        ← 对应 clean 输入的输出目录
    │   └── test/         ← 对应 test 输入的输出目录


    ├── realtimeout_clean/
    ├── realtimeout_test/
    ├── test/
    ├── exp-1_lg_clean_1.txt
    └── exp-1_lg_test_1.txt
```
如果运行失败，检查18000端口是否被占用

等待开始运行后自动进入前端界面

由于是现实时间模拟，所以运行时间和输入音频总时长相同

完毕后自动跳转到后端窗口显示

`INFO:connection closed`

手动`ctrl c`退出后端


### 结果检查

运行成功后，在任意终端运行指令：
```
for d in exp/exp-1/HD-Track2/*; do echo "$(basename "$d"): $(find "$d" -maxdepth 1 -type f -name "*.wav" | wc -l)"; done
```
查看是否和输入文件数目相同，进行正确性检验


