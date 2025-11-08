import os
import wave
import subprocess
import numpy as np
import sherpa_onnx
import openai
from piper import PiperVoice
from pathlib import Path
import os
import re
import requests
import base64
import json
import tempfile
import soundfile as sf

root_path = "model"

ASR_MODEL = sherpa_onnx.OnlineRecognizer.from_transducer(
    encoder=f"{root_path}/sherpa-onnx-streaming-zipformer-small-bilingual-zh-en-2023-02-16/encoder-epoch-99-avg-1.onnx",
    decoder=f"{root_path}/sherpa-onnx-streaming-zipformer-small-bilingual-zh-en-2023-02-16/decoder-epoch-99-avg-1.onnx",
    joiner=f"{root_path}/sherpa-onnx-streaming-zipformer-small-bilingual-zh-en-2023-02-16/joiner-epoch-99-avg-1.onnx",
    tokens=f"{root_path}/sherpa-onnx-streaming-zipformer-small-bilingual-zh-en-2023-02-16/tokens.txt",
    num_threads=1,
)
VOICE = PiperVoice.load(f"{root_path}/tts/zh_CN-huayan-medium.onnx")
OPENAI_CLIENT = openai.OpenAI(api_key=os.getenv("LLM_API_KEY", "not-needed"), base_url="http://127.0.0.1:8000/v1")
LLM_MODEL = "Qwen2.5-0.5B-Instruct"
LLM_SYSTEM = "你是客服机器人，几个字回答用户,文本不要有任何格式"

QWEN_URL = "http://127.0.0.1:10004/v1/chat/completions"


def asr(path):
    stream = ASR_MODEL.create_stream()
    with wave.open(path, "rb") as f:
        sr = f.getframerate()
        audio = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    stream.accept_waveform(sr, audio)
    stream.input_finished()
    while ASR_MODEL.is_ready(stream):
        ASR_MODEL.decode_stream(stream)
    result = ASR_MODEL.get_result(stream)
    text = result.text if hasattr(result, "text") else result
    return str(text).strip()


def llm(text):
    completion = OPENAI_CLIENT.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": LLM_SYSTEM},
            {"role": "user", "content": text},
        ],
        max_tokens=64,
    )
    return completion.choices[0].message.content.strip()

def llm_qwen3o(prompt: str, audio_array: np.ndarray = None, sr: int = 16000):
    """
    调用 Qwen3-Omni 多模态接口。
    参数：
        prompt: 用户的文本提示（必须）
        audio_array: 可选的音频 ndarray(float32)，范围[-1, 1]
        sr: 采样率，默认 16000
    返回：
        Qwen 返回的文本字符串
    """
    # messages = [{"role": "system", "content": "你是一个语音客服,你要没有任何格式的在50字10s左右回答用户"}]
    messages = [{"role": "system", "content": "你是一个语音客服,你要没有任何格式的在10个字左右回答用户"}]

    # 如果包含音频，构造音频+文本混合输入
    if audio_array is not None:
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmpfile:
                sf.write(tmpfile.name, audio_array, sr)
                tmpfile.seek(0)
                audio_base64 = base64.b64encode(tmpfile.read()).decode("utf-8")
            messages.append({
                "role": "user",
                "content": [
                    {"type": "audio_url", "audio_url": {"url": f"data:audio/wav;base64,{audio_base64}"}},
                    {"type": "text", "text": prompt}
                ]
            })
        except Exception as e:
            print(f"[QWEN AUDIO ENCODE ERROR] {e}")
            return ""
    else:
        messages.append({"role": "user", "content": [{"type": "text", "text": prompt}]})

    payload = {
        "temperature": 1.0,
        "top_p": 0.7,
        "top_k": 40,
        "presence_penalty": 1.2,
        "frequency_penalty": 0.8,
        "max_tokens": 256,
        "seed": 42,
        "messages": messages
    }

    try:
        response = requests.post(
            QWEN_URL,
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload),
            timeout=300
        )
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        return content
    except Exception as e:
        print(f"[QWEN REQUEST ERROR] {e}")
        return ""


def tts(text, path):
    with wave.open(str(path), "wb") as f:
        VOICE.synthesize_wav(text, f)
    return str(path)



def get_wav(input_dir="/home/sds/data", mode="time"):
    """
    根据模式返回指定目录下的 wav 文件列表。
    参数:
        input_dir (str): 音频所在目录。
        mode (str):
            - 'time'  表示处理带打断音频（排除 clean_ 开头）
            - 'clean' 表示处理干净音频（只取 clean_ 开头）
            - 'all'   表示读取所有不以 'output.wav' 结尾的音频文件

    返回:
        list[str]: 符合条件的 wav 文件名列表。
    """
    wav_files = []
    for f in os.listdir(input_dir):
        if not f.endswith(".wav"):
            continue

        # --- 模式1：time（排除clean_）
        if mode == "time":
            if f.lower().startswith("clean"):
                continue
            if re.match(r"^\d", f):  # 例如 0001_0002.wav
                wav_files.append(f)

        # --- 模式2：clean（只取clean_）
        elif mode == "clean":
            if f.lower().startswith("clean_"):
                wav_files.append(f)

        # --- 模式3：all（排除output.wav）
        elif mode == "all":
            if not f.lower().endswith("output.wav"):
                wav_files.append(f)

    wav_files.sort()
    return wav_files


def api_qwen3o(prompt: str, audio_array: np.ndarray = None, sr: int = 16000) -> str:
    """
    ✅ DashScope / Qwen3-Omni 兼容模式输入
    支持文字 + 音频（base64 dataURL 格式）输入
    """
    client = openai.OpenAI(
        api_key="sk-553f353ca3d1436b9ec9a9c728e30958",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    messages = [{"role": "system", "content": "你是一个语音客服,请在10个字以内自然回答用户"}]

    if audio_array is not None:
        try:
            # 将音频数组写入临时文件
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
                sf.write(tmp.name, audio_array, sr)
                tmp.seek(0)
                audio_b64 = base64.b64encode(tmp.read()).decode("utf-8")

            # 必须加上 "data:audio/wav;base64," 前缀
            data_url = f"data:audio/wav;base64,{audio_b64}"

            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": data_url,
                            "format": "wav"
                        }
                    },
                    {"type": "text", "text": prompt}
                ]
            })
        except Exception as e:
            print(f"[QWEN AUDIO ENCODE ERROR] {e}")
            return ""
    else:
        messages.append({
            "role": "user",
            "content": [{"type": "text", "text": prompt}]
        })

    # === 发起流式请求 ===
    completion = client.chat.completions.create(
        model="qwen3-omni-flash",
        messages=messages,
        modalities=["text"],        # 这里只要返回文字即可
        audio={"voice": "Cherry", "format": "wav"},
        stream=True,
        stream_options={"include_usage": True},
        extra_body={"enable_thinking": False},  # 禁用思考模式，防止音频无响应
    )

    text_output = ""
    try:
        for chunk in completion:
            if chunk.choices:
                delta = chunk.choices[0].delta
                if hasattr(delta, "content") and delta.content:
                    text_output += delta.content
    except Exception as e:
        print(f"[QWEN STREAM ERROR] {e}")

    return text_output.strip()

def main():
    base_dir = Path("test_wav")


if __name__ == "__main__":
    main()