cd model
git clone https://github.com/Ksuriuri/index-tts-vllm.git
cd index-tts-vllm
modelscope download --model kusuriuri/Index-TTS-1.5-vLLM --local_dir ./checkpoints/Index-TTS-1.5-vLLM
conda create -n index-tts-vllm python=3.12
conda activate index-tts-vllm
pip install -r requirements.txt