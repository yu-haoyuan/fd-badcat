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


def llm_qwen3o(messages: list):
    payload = {
        "temperature": 0,
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
        return data["choices"][0]["message"]["content"]

    except Exception as e:
        print(f"[QWEN REQUEST ERROR] {e}")
        return ""


def main():
    base_dir = Path("test_wav")


if __name__ == "__main__":
    main()