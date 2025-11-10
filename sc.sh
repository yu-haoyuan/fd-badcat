# CUDA_VISIBLE_DEVICES=5 LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.550.54.14:/usr/lib/x86_64-linux-gnu/libcuda.so.550.54.14 python model/index-tts-vllm/api_server.py

# # ---------- 4. 执行 inter_score.py ----------
# echo "运行 inter_score.py ..."
# python test_path_tts/inter_score.py

# # ---------- 5. 执行 inter_sum.py ----------
echo "运行 inter_sum.py ..."
python test_path_tts/inter_sum.py

# ---------- 6. 执行 ave.py ----------
echo "运行 ave.py ..."
python test_path_tts/ave.py