# fd-badcat
full duplex-spoken dialogue system

---
### 环境准备
我们提供了
```
docker build --progress=plain -t fd-badcat .
```
但是请注意，我们出现了多次在配置qwen3omni环境的vllm编译阶段的不可避免的错误，所以如果`docker file`报错请根据以下步骤安装环境

```
bash setup/qwen3omni_env.sh
bash setup/indextts_env.sh
bash setup/aux_model.sh
```

准备完毕后正确的文件目录为
```
model/
├── Qwen3-Omni-30B-A3B-Instruct/
├── index-tts-vllm/
│   └── checkpoints/
│       └── Index-TTS-1.5-vLLM/
└── sherpa-onnx-paraformer-zh-2024-03-09/
```

正确的环境内容为
```
conda env list
# conda environments:
fd-sds                   /root/miniconda3/envs/fd-sds(系统运行环境)
index-tts-vllm           /root/miniconda3/envs/index-tts-vllm(index服务环境)
fdbc-qwen3o-vllm         /root/miniconda3/envs/vllm(qwen3omni环境)

```


### 数据准备

创建`exp`文件夹

```
mkdir exp/exp-1
```
然后将`test/clean`放到`exp/exp-1`下面
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

我们的实验采取模拟现实时长的前后端模式，所以需要`screen`或者`tmux`进行多终端运行,但是如果实验足够短，那么简单的多终端运行也是可以接受的

所以经过我们的测试，手动在多终端启动是便捷可靠的

请在终端1
```
conda activate fdbc-qwen3o-vllm 
vllm serve model/Qwen3-Omni-30B-A3B-Instruct --port 10003 --host 0.0.0.0 --dtype bfloat16 --max-model-len 65536 --allowed-local-media-path / -tp 4
```
启动qwen3omni vllm模型，如果正确启动，会在该终端下面看到
`0.0.0.0:10003`的启动说明


请在终端2
```
conda activate index-tts-vllm
python model/index-tts-vllm/api_server.py

```
启动qwen3omni vllm模型，如果正确启动，会在该终端下面看到
`0.0.0.0:10003`的启动说明

##### 2.主实验启动
如果测试数据集文件夹遵循测试集-test/clean格式，对应脚本文件夹为`./src`

关于dev得分和实验结果的脚本在`./exp-dev`

启动实验脚本之前，确认本机是否安装tmux
```
command -v tmux >/dev/null 2>&1 || (sudo apt update && sudo apt install -y tmux)
bash src/sc.sh
```

如果脚本一键运行失败

那么请在不同的终端手动启动前后端脚本
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
    │   ├── clean/
    │   └── test/
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

然后运行
```
for d in exp/exp-1/HD-Track2/*; do echo "$(basename "$d"): $(find "$d" -maxdepth 1 -type f -name "*.wav" | wc -l)"; done
```
查看是否和输入文件数相同，进行正确性检验


