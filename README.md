# fd-badcat
fd-sds

---
dabu 1108

screen -S index_vllm

conda activate index

port = 19000

CUDA_VISIBLE_DEVICES=5 LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.550.54.14:/usr/lib/x86_64-linux-gnu/libcuda.so.550.54.14 python model/index-tts-vllm/api_server.py

---

server

screen -S zh_server

conda activate sds

python test_path_tts/ws_pipe.py \
--medium "realtime_out1" \
--port 18000 # zh

screen -S zh_client

conda activate sds

python test_path_tts/ws_f.py \
--exp "exp4" \
--lang "zh" \
--port 18000 # zh

---

server

screen -S en_server

conda activate sds

python test_path_tts/ws_pipe.py \
--medium "realtime_out2" \
--port 18001 # en

screen -S en_client

conda activate sds

python test_path_tts/ws_f.py \
--exp "exp4" \
--lang "en" \
--port 18001 # en


