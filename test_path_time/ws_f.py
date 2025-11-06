# -*- coding: utf-8 -*-
"""
simulate_full_frontend.py （时间戳同步版）
------------------------------------------
- 🎙️ 从本地 WAV 模拟麦克风输入，每16ms发送一帧
- 🔊 收到后端返回的 TTS 音频，根据其 timestamp 插入
- 🕒 时间同步：与后端 start_wall 时间保持一致
------------------------------------------
"""

import asyncio
import json
import io
import time
import soundfile as sf
import numpy as np
import websockets
from pathlib import Path


# ========== 基本配置 ==========
WS_URL = "ws://127.0.0.1:18000/realtime"
INPUT_WAV = "exp/exp2/0001_0003.wav"
SAMPLE_RATE = 16000
CHUNK_SAMPLES = 256  # 16 ms per frame
# ==============================


class SpeakerSimulator:
    def __init__(self, total_duration: float, sr=SAMPLE_RATE, output_path: Path = None):
        self.sr = sr
        self.chunk_time = CHUNK_SAMPLES / sr
        self.total_samples = int(total_duration * sr)
        self.output_path = output_path

        # 初始化全静音 buffer (float32)
        self.audio_buffer = np.zeros(self.total_samples, dtype=np.float32)
        self.start_wall = None  # 后端计时参考起点
        self.interrupted = False
        print(f"🧮 初始化输出 buffer，总时长 {total_duration:.2f}s ({self.total_samples} samples)")

    def reset_for_new_audio(self, wav_bytes: bytes, start_time: float):
        """在指定时间戳位置插入 TTS 音频"""
        data, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
        assert sr == self.sr, f"采样率不匹配: {sr}"

        start_sample = int(start_time * self.sr)
        available = self.total_samples - start_sample
        write_samples = min(len(data), max(0, available))

        if write_samples > 0:
            self.audio_buffer[start_sample:start_sample + write_samples] = data[:write_samples]
            print(f"🎵 在 {start_time:.2f}s 插入 TTS（{write_samples / self.sr:.2f}s）")

    def handle_interrupt(self):
        """收到打断后，后续保持静音"""
        self.interrupted = True
        print("🛑 播放被打断，后续输出静音")

    def save_output(self):
        """保存最终输出音频"""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(self.output_path, self.audio_buffer, self.sr)
        print(f"💾 已保存对齐音频: {self.output_path}, 时长 {len(self.audio_buffer) / self.sr:.2f}s")


async def mic_sender(ws, wav_path: Path):
    """实时发送音频帧"""
    data, sr = sf.read(str(wav_path), dtype="float32")
    if sr != SAMPLE_RATE:
        raise ValueError(f"采样率不匹配: {sr}")

    frame_time = CHUNK_SAMPLES / sr
    total_frames = int(np.ceil(len(data) / CHUNK_SAMPLES))
    print(f"🎙️ 开始发送 {wav_path.name}, 总帧 {total_frames}，时长 {len(data)/sr:.2f}s")
    t0 = time.perf_counter()

    for i in range(0, len(data), CHUNK_SAMPLES):
        chunk = data[i:i + CHUNK_SAMPLES]
        if len(chunk) < CHUNK_SAMPLES:
            chunk = np.pad(chunk, (0, CHUNK_SAMPLES - len(chunk)))
        await ws.send(chunk.tobytes())
        await asyncio.sleep(frame_time)

    await ws.send(json.dumps({"event": "end"}))
    print(f"📤 音频发送完毕，用时 {time.perf_counter() - t0:.2f}s")


async def simulate_full_frontend():
    input_path = Path(INPUT_WAV)
    output_path = input_path.parent / f"{input_path.stem}_output.wav"

    # 获取总时长
    data, sr = sf.read(str(input_path), dtype="float32")
    total_duration = len(data) / sr

    async with websockets.connect(WS_URL, max_size=None) as ws:
        print(f"✅ 已连接后端: {WS_URL}")
        speaker = SpeakerSimulator(total_duration, output_path=output_path)

        # === 状态变量 ===
        last_tts_timestamp = None  # 后端 tts_done timestamp（秒）

        # 启动麦克风发送
        send_task = asyncio.create_task(mic_sender(ws, input_path))

        # === 接收协程 ===
        async def receiver():
            nonlocal last_tts_timestamp

            while True:
                try:
                    msg = await ws.recv()
                except websockets.exceptions.ConnectionClosed:
                    print("⚠️ WebSocket 已关闭，结束接收循环")
                    break

                if isinstance(msg, bytes):
                    if last_tts_timestamp is None:
                        print("⚠️ 收到音频但无 timestamp，跳过写入")
                        continue
                    start_time = last_tts_timestamp
                    speaker.reset_for_new_audio(msg, start_time)
                    last_tts_timestamp = None  # 用一次即清空
                    continue

                try:
                    obj = json.loads(msg)
                    event = obj.get("event")

                    if event == "tts_done":
                        last_tts_timestamp = obj["data"].get("timestamp")
                        print("tts:", obj)
                        print(f"🕒 收到 tts_done, timestamp={last_tts_timestamp}s")

                    elif event == "stop_audio":
                        speaker.handle_interrupt()

                    else:
                        print("📨 其他消息:", obj)

                except Exception:
                    print("📨 文本消息:", msg)

                await asyncio.sleep(0)  # 释放控制权

        recv_task = asyncio.create_task(receiver())

        await send_task  # 等待音频发送完毕
        speaker.save_output()  # ✅ 立即保存输出


if __name__ == "__main__":
    asyncio.run(simulate_full_frontend())
