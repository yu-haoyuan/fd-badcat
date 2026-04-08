import os
from pathlib import Path
import requests
import json
import soundfile as sf
import io
import numpy as np
import torch
import torchaudio
import dashscope
from openai import OpenAI

# DashScope API Key 配置（三个模型共用）
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
if not DASHSCOPE_API_KEY:
    raise ValueError("请设置环境变量 DASHSCOPE_API_KEY")

# 初始化 Omni 模型客户端
omni_client = OpenAI(
    api_key=DASHSCOPE_API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

def tts(text, path):
    """
    使用 qwen3-tts-flash 进行文本转语音
    输入: text (str) - 待转换文本, path (Path/str) - 输出音频路径
    输出: str - 音频文件路径
    """
    print(f"[TTS] 开始生成语音: {text[:50]}...")
    
    response = dashscope.MultiModalConversation.call(
        model="qwen3-tts-flash",
        api_key=DASHSCOPE_API_KEY,
        text=text,
        voice="Cherry",
        language_type="Chinese",
        stream=False
    )
    
    if response.status_code != 200:
        raise Exception(f"TTS API 调用失败: {response.message}")
    
    # 获取音频 URL
    audio_url = response.output.audio.url
    if not audio_url:
        raise Exception("TTS 返回的音频 URL 为空")
    
    print(f"[TTS] 下载音频: {audio_url[:80]}...")
    
    # 从 URL 下载音频文件
    audio_response = requests.get(audio_url)
    audio_response.raise_for_status()
    audio_data = audio_response.content
    
    # 读取音频数据并转换为 16kHz
    data, sr = sf.read(io.BytesIO(audio_data), dtype="float32")
    
    if sr != 16000:
        data = torchaudio.functional.resample(
            torch.from_numpy(data).unsqueeze(0), sr, 16000
        ).squeeze(0).numpy()
        sf.write(str(path), data, 16000, subtype="PCM_16")
    else:
        sf.write(str(path), data, sr, subtype="PCM_16")
    
    print(f"[TTS] 语音生成完成: {path}")
    return str(path)

def asr(path):
    """
    使用 qwen3-asr-flash 进行语音识别
    输入: path (str/Path) - 音频文件路径
    输出: str - 识别出的文本
    """
    print(f"[ASR] 开始识别音频: {path}")
    
    # 读取本地音频文件并上传到临时位置（使用 file:// 协议或 base64）
    # DashScope ASR API 支持本地文件路径
    audio_path = str(path)
    
    messages = [
        {
            "role": "system",
            "content": [
                {"text": ""},
            ]
        },
        {
            "role": "user",
            "content": [
                {"audio": audio_path},
            ]
        }
    ]
    
    response = dashscope.MultiModalConversation.call(
        api_key=DASHSCOPE_API_KEY,
        model="qwen3-asr-flash",
        messages=messages,
        result_format="message",
        asr_options={
            "enable_lid": True,
            "enable_itn": False
        }
    )
    
    if response.status_code != 200:
        raise Exception(f"ASR API 调用失败: {response.message}")
    
    # 提取识别结果
    text = response.output.choices[0].message.content[0]["text"]
    print(f"[ASR] 识别完成: {text}")
    return str(text).strip()

def llm_qwen3o(messages: list):
    """
    使用 qwen3.5-omni-flash 进行对话（支持文本和音频输出）
    输入: messages (list) - 对话消息列表
    输出: str - 回复的文本内容
    """
    print(f"[Omni] 开始调用 qwen3.5-omni-flash")
    
    try:
        completion = omni_client.chat.completions.create(
            model="qwen3-omni-flash-2025-12-01",
            messages=messages,
            modalities=["text"],
            stream=True,
            stream_options={"include_usage": True},
        )
        
        text_content = ""
        for chunk in completion:
            if chunk.choices:
                delta = chunk.choices[0].delta
                # 提取文本内容
                if hasattr(delta, 'content') and delta.content:
                    text_content += delta.content
        
        print(f"[Omni] 响应完成")
        return text_content
        
    except Exception as e:
        print(f"[Omni REQUEST ERROR] {e}")
        return ""

def main():
    base_dir = Path("test_wav")
if __name__ == "__main__":
    main()