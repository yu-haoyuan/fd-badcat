# fd-badcat
full duplex-spoken dialogue system

> [Unit-Based Agent for Semi-Cascaded Full-Duplex Dialogue Systems](https://arxiv.org/abs/2601.20230) <br>
> [Haoyuan Yu](https://yu-haoyuan.github.io/), [Yuxuan Chen], [Minjie Cai](https://cai-mj.github.io/) <br>
> ICASSP 2026 Grand Challenge
---

Our paper is accepted by **ICASSP-2026 Grand Challenge**

![image](https://github.com/yu-haoyuan/fd-badcat/blob/main/fig.png)


---

### Environment Preparation

We provide a one-click startup Docker environment:

```
docker build --progress=plain -t fd-badcat .

```

However, please note that due to issues with domestic Docker mirrors in China, we encountered unavoidable errors multiple times during the vLLM compilation stage during trial runs. Therefore, if the `docker file` throws an error, please follow the steps below to install the environment manually (we have confirmed locally that the following solution is 100% viable):

First, confirm whether `tmux` is installed on the machine:

```
command -v tmux >/dev/null 2>&1 || (sudo apt update && sudo apt install -y tmux)

```

Then, in the terminal, run the following scripts in order:

```
bash setup/qwen3omni_env.sh
bash setup/indextts_env.sh
bash setup/aux_model.sh

```

---

### Environment Check

After completing the installation via Docker or scripts, use `conda env list` to check. The correct environment content should be:

```
fd-sds                   /root/miniconda3/envs/fd-sds (System runtime environment)
index-tts-vllm           /root/miniconda3/envs/index-tts-vllm (Index service environment)
fdbc-qwen3o-vllm         /root/miniconda3/envs/vllm (Qwen3Omni environment)

```

Once prepared, the correct directory structure for the `model` subfolder is:

```
model/
├── Qwen3-Omni-30B-A3B-Instruct/
├── index-tts-vllm/
│   └── checkpoints/
│       └── Index-TTS-1.5-vLLM/
└── sherpa-onnx-paraformer-zh-2024-03-09/

```

---

### Data Preparation

Create the `exp/exp-1` folder as the designated data directory:

```
mkdir exp/exp-1

```

Then, place the `test/clean` directories that meet the competition requirements under `exp/exp-1`:

```
exp/
└── exp-1/
    ├── clean/
    └── test/

```

---

### Startup Instructions

##### 1. API Startup

Our repository is primarily based on calling the `qwen3omni` API and the `indextts-1.5` API for experiments.

The logic of our project is relatively simple. If the two APIs are configured correctly, the environment dependencies of the experiment itself will not cause issues, as it only relies on basic frontend and backend tools.

Our experiment adopts a frontend-backend mode that simulates real-time duration. This means that the length of the raw data ≈ the duration of the experiment run. Therefore, `screen` or `tmux` is required for continuous concurrent operation across multiple terminals.

However, if the experiment is short enough, simple multi-terminal execution is also acceptable. Based on our testing, manually starting in multiple terminals is convenient and reliable (thanks to vLLM's concurrency optimization).

**The experiment must be run with all five terminals successfully started.**

In **Terminal 1**, run the following command:

```
conda activate fdbc-qwen3o-vllm 
vllm serve model/Qwen3-Omni-30B-A3B-Instruct --port 10003 --host 0.0.0.0 --dtype bfloat16 --max-model-len 65536 --allowed-local-media-path / -tp 4

```

This starts the Qwen3Omni vLLM model. If started correctly, you will see `running on http://0.0.0.0:10003` in the terminal. Please keep this terminal open.

In **Terminal 2**, run the following command:

```
conda activate fdbc-qwen3o-vllm 
python src/qwen3_api.py

```

If started correctly, you will see `running on http://0.0.0.0:10004`. Please keep this terminal open.

In **Terminal 3**, run the following command:

```
conda activate index-tts-vllm
python model/index-tts-vllm/api_server.py

```

This starts the Index-TTS vLLM model. If started correctly, you will see `INFO: Uvicorn running on http://0.0.0.0:19000 (Press CTRL+C to quit)`. Please keep this terminal open as well.

##### 2. Main Experiment Startup

Run the script directly:

```
bash src/sc.sh

```

This will automatically launch the frontend and backend and begin synthesizing output. It will prompt `Startup Complete`. At this point, there is still 1 physical terminal with 2 tmux windows running the services.

If the `sc.sh` one-click script fails, please manually start the frontend and backend scripts in **Terminals 4 and 5**:

```
python src/backend.py --config fd-badcat/src/config.yaml
python src/frontend.py --config fd-badcat/src/config.yaml

```

The final correct output structure will be:

```
exp/
└── exp-1/
    ├── clean/
    ├── HD-Track2/         ← This is the folder for output
    │   ├── clean/         ← Output directory corresponding to clean input
    │   └── test/          ← Output directory corresponding to test input
    ├── realtimeout_clean/
    ├── realtimeout_test/
    ├── test/
    ├── exp-1_lg_clean_1.txt
    └── exp-1_lg_test_1.txt

```

If the run fails, check if port 18000 is occupied.

Once the run starts, it will automatically enter the frontend interface.

Since this is a real-time simulation, the execution time is equal to the total duration of the input audio.

Upon completion, it will automatically jump to the backend window displaying:
`INFO:connection closed`

Manually press `Ctrl+C` to exit the backend.

### Results Check

After a successful run, execute the following command in any terminal:

```
for d in exp/exp-1/HD-Track2/*; do echo "$(basename "$d"): $(find "$d" -maxdepth 1 -type f -name "*.wav" | wc -l)"; done

```

Verify if the number of files matches the input file count to validate correctness.

