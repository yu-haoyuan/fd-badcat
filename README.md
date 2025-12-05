# fd-badcat
full duplex-spoken dialogue system

---
### 模型以及环境准备

需要的模型分别有

`qwen3omni `

`index tts`

`sherpa-onnx-paraformer-zh-2024-03-09`

在三个终端中分别运行
```
bash setup/qwen3o_api.sh
bash setup/index_api.sh
bash setup/aux_model.sh
```

自动安装环境

准备完毕后正确的文件目录为
```
/fd-badcat/model
├── Index-TTS-1.5-vLLM
├── index-tts-vllm
├── qwen3omni
├── sherpa-onnx-paraformer-zh-2024-03-09
├── vllm
└── vllm_env.tar.gz
```

正确的环境内容为
```
conda env list
# conda environments:
#
fd-sds                   /root/miniconda3/envs/fd-sds(系统运行环境)
index-tts-vllm           /root/miniconda3/envs/index-tts-vllm(index服务环境)
vllm                     /root/miniconda3/envs/vllm(qwen3omni环境)

```

### qwen3o失败情况下环境安装方式
```
这里要解压到自己的conda环境下面,如我这里是/root/miniconda3
mkdir -p /root/miniconda3/envs/vllm
tar -xzvf vllm_env.tar.gz -C /root/miniconda3/envs/vllm --strip-components=1
source /root/miniconda3/etc/profile.d/conda.sh
conda activate vllm
cd model/vllm
export VLLM_PRECOMPILED_WHEEL_LOCATION="https://wheels.vllm.ai/a5dd03c1ebc5e4f56f3c9d3dc0436e9c582c978f/vllm-0.9.2-cp38-abi3-manylinux1_x86_64.whl"
VLLM_USE_PRECOMPILED=1 pip install -e . -v --no-build-isolation

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
如果文件夹遵循测试集-test/clean格式，对应脚本文件夹为`./src`

关于dev得分和实验结果的脚本在`./exp-dev`

启动
```
bash src/sc.sh
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



