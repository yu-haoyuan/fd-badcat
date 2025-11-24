# fd-badcat
fd-sds

---
### 模型准备

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

### 数据准备

创建`exp`文件夹

```
mkdir -p exp/exp-1
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
