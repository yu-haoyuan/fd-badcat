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
import io, torch
import torchaudio

asr_dir = "model/sherpa-onnx-paraformer-zh-2024-03-09"
ASR_MODEL = sherpa_onnx.OfflineRecognizer.from_paraformer(
    paraformer=f"{asr_dir}/model.onnx",
    tokens=f"{asr_dir}/tokens.txt",
    num_threads=2,
    provider="cpu",  # 可改为 "cuda" 使用 GPU
)

OPENAI_CLIENT = openai.OpenAI(api_key=os.getenv("LLM_API_KEY", "not-needed"), base_url="http://127.0.0.1:8000/v1")

LLM_SYSTEM = "你是客服机器人，几个字回答用户,文本不要有任何格式,根据用户语言决定你回答的语言"

QWEN_URL = "http://127.0.0.1:10004/v1/chat/completions"


def _call_index_tts(text: str) -> bytes:
    url = "http://127.0.0.1:19000/tts"
    payload = {
        "text": text,
        "character": "ht"
    }
    resp = requests.post(url, json=payload)
    resp.raise_for_status()
    return resp.content

def tts(text, path):
    print("tts text")
    wav_bytes = _call_index_tts(text)

    data, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
    if sr != 16000:
        data = torchaudio.functional.resample(
            torch.from_numpy(data).unsqueeze(0), sr, 16000
        ).squeeze(0).numpy()
        sf.write(path, data, 16000, subtype="PCM_16")
    else:
        sf.write(path, data, sr, subtype="PCM_16")

    return str(path)

def asr(path):
    audio, sr = sf.read(path, dtype="float32")
    stream = ASR_MODEL.create_stream()
    stream.accept_waveform(sr, audio)
    ASR_MODEL.decode_stream(stream)
    print("asrok")
    return str(stream.result.text).strip()


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
    messages = [{"role": "system", "content": "你是一个语音客服,你要没有任何格式的在10个字左右回答用户，根据用户语言决定你回答的语言，用户语言是中文的时候，回答中文，用户语言是英文的时候，回答英文"}]

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


def main():
    base_dir = Path("test_wav")


if __name__ == "__main__":
    main()