# -*- coding: utf-8 -*-
"""
simulate_full_frontend.py
------------------------------------------
功能：
- 🎙️ 模拟麦克风输入：从本地 WAV 文件，每16ms发送一帧到后端。
- 🔊 模拟喇叭播放：按时间轴填充 TTS 音频或静音，输出与输入等长的 WAV。
- 🕒 真正按实时节奏运行：前后端可同步评测。
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
    def __init__(self, total_frames: int, sr=SAMPLE_RATE, output_path: Path = None):
        self.sr = sr
        self.chunk_time = CHUNK_SAMPLES / sr
        self.total_frames = total_frames
        self.total_samples = total_frames * CHUNK_SAMPLES
        self.output_path = output_path

        # 初始化全静音 buffer (float32, shape: [total_samples])
        self.audio_buffer = np.zeros(self.total_samples, dtype=np.float32)
        self.current_frame = 0  # 当前时间进度（帧数，0-indexed）
        self.playing_tts_until = -1  # 当前正在播放的 TTS 覆盖到哪一帧（inclusive）
        self.interrupted = False   # 全局打断标志

    def reset_for_new_audio(self, wav_bytes: bytes, start_frame: int):
        """在指定帧位置插入 TTS 音频"""
        data, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
        assert sr == self.sr, f"采样率不匹配: {sr}"

        # 计算能写入多少 samples（避免越界）
        start_sample = start_frame * CHUNK_SAMPLES
        available = self.total_samples - start_sample
        write_samples = min(len(data), available)

        if write_samples > 0:
            self.audio_buffer[start_sample : start_sample + write_samples] = data[:write_samples]
            end_frame = start_frame + int(np.ceil(write_samples / CHUNK_SAMPLES))
            self.playing_tts_until = min(end_frame - 1, self.total_frames - 1)
            print(f"🎵 在帧 {start_frame} 插入 TTS，持续 {write_samples/self.sr:.2f}s")

    def handle_interrupt(self):
        """收到打断后，后续保持静音（已写入的不变）"""
        self.interrupted = True
        print("🛑 播放被打断，后续输出静音")

    def advance_frame(self):
        """每 16ms 调用一次，推进时间轴"""
        if self.current_frame < self.total_frames:
            self.current_frame += 1

    def save_output(self):
        """保存完整时长的输出音频"""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(self.output_path, self.audio_buffer, self.sr)
        actual_duration = len(self.audio_buffer) / self.sr
        print(f"💾 已保存对齐音频: {self.output_path}, 时长 {actual_duration:.2f}s")


async def mic_sender(ws, wav_path: Path):
    data, sr = sf.read(str(wav_path), dtype="float32")
    if sr != SAMPLE_RATE:
        raise ValueError(f"采样率不匹配: {sr}")
    total_samples = len(data)
    total_frames = int(np.ceil(total_samples / CHUNK_SAMPLES))
    frame_time = CHUNK_SAMPLES / sr
    print(f"🎙️ 开始发送 {wav_path.name}, 总时长 {total_samples/sr:.2f}s ({total_frames} 帧)")
    t0 = time.perf_counter()

    for i in range(0, total_samples, CHUNK_SAMPLES):
        chunk = data[i:i+CHUNK_SAMPLES]
        if len(chunk) < CHUNK_SAMPLES:
            pad = np.zeros(CHUNK_SAMPLES - len(chunk), dtype=np.float32)
            chunk = np.concatenate([chunk, pad])
        await ws.send(chunk.tobytes())
        await asyncio.sleep(frame_time)
    await ws.send(json.dumps({"event": "end"}))
    print(f"📤 音频发送完毕，用时 {time.perf_counter()-t0:.2f}s")
    return total_frames


async def simulate_full_frontend():
    input_path = Path(INPUT_WAV)
    output_path = input_path.parent / f"{input_path.stem}_output.wav"

    # 先读取输入音频，计算总帧数
    data, sr = sf.read(str(input_path), dtype="float32")
    if sr != SAMPLE_RATE:
        raise ValueError(f"输入音频采样率不匹配: {sr}")
    total_samples = len(data)
    total_frames = int(np.ceil(total_samples / CHUNK_SAMPLES))

    async with websockets.connect(WS_URL, max_size=None) as ws:
        print(f"✅ 已连接后端: {WS_URL}")
        speaker = SpeakerSimulator(total_frames=total_frames, output_path=output_path)

        # 启动发送任务，获取总帧数（其实已提前算好）
        send_task = asyncio.create_task(mic_sender(ws, input_path))

        # 接收消息
        async def receiver():
            while speaker.current_frame < total_frames:
                try:
                    msg = await ws.recv()
                except websockets.exceptions.ConnectionClosed:
                    break

                if isinstance(msg, bytes):
                    # 收到 TTS 音频：从当前帧开始插入
                    speaker.reset_for_new_audio(msg, speaker.current_frame)
                else:
                    try:
                        obj = json.loads(msg)
                        print("📨 收到JSON消息:", obj)
                        if obj.get("event") == "stop_audio":
                            speaker.handle_interrupt()
                    except Exception:
                        print("📨 文本消息:", msg)

                # 每收到一条消息（或超时）推进一帧？不，我们按真实时间推进
                # 改为：由主循环控制时间推进
                await asyncio.sleep(0)  # 让出控制权

        recv_task = asyncio.create_task(receiver())

        # === 主时间循环：每 16ms 推进一帧 ===
        frame_time = CHUNK_SAMPLES / SAMPLE_RATE
        for frame_idx in range(total_frames):
            speaker.advance_frame()
            await asyncio.sleep(frame_time)

        # 等待任务结束
        await asyncio.gather(send_task, recv_task, return_exceptions=True)

        # 保存最终结果
        speaker.save_output()


if __name__ == "__main__":
    asyncio.run(simulate_full_frontend())