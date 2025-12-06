import requests

SERVER_PORT = 6006

# 1. 使用本地文件请求，将 audio_paths 中的路径改为本地音频文件
url = F"http://0.0.0.0:{SERVER_PORT}/tts_url"
data = {
    "text": "还是会想你，还是想登你",
    "audio_paths": [  # 支持多参考音频
        "assets/jay_promptvn.wav",
        "assets/vo_card_klee_endOfGame_fail_01.wav"
    ]
}

response = requests.post(url, json=data)
with open("output.wav", "wb") as f:
    f.write(response.content)



# 2. 使用角色名请求，角色注册请参考 `assets/speaker.json`
url = f"http://0.0.0.0:{SERVER_PORT}/tts"
data = {
    "text": "还是会想你，还是想登你",
    "character": "jay_klee"
}

response = requests.post(url, json=data)
with open("output.wav", "wb") as f:
    f.write(response.content)