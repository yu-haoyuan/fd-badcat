# CUDA_VISIBLE_DEVICES=5 LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.550.54.14:/usr/lib/x86_64-linux-gnu/libcuda.so.550.54.14 python model/index-tts-vllm/api_server.py

# echo "启动 ws_pipe.py (后端监听) ..."
# python test_path_tts/ws_pipe.py &
# PIPE_PID=$!

# sleep 10  # 确保 WebSocket 服务已准备好

# # ---------- 3. 启动 ws_f.py （前端）----------
# echo "启动 ws_f.py (前端模拟) ..."
python test_path_tts/ws_f.py --config test_path_tts_p/config.yaml

#注意：ws_f.py 是前端脚本，执行完毕后会退出（打印 “全部完成”）。
# 此时后端 ws_pipe.py 仍在后台运行。

# ---------- 4. 执行 inter_score.py ----------
echo "运行 inter_score.py ..."
python test_path_tts/inter_score.py --config test_path_tts_p/config.yaml

# ---------- 5. 执行 inter_sum.py ----------
echo "✅ 运行 inter_sum.py ..."
python test_path_tts/inter_sum.py --config test_path_tts_p/config.yaml

# ---------- 6. 执行 ave.py ----------
echo "✅ 运行 ave.py ..."
python test_path_tts/ave.py --config test_path_tts_p/config.yaml

# ---------- 7. 结束后台进程 ----------


