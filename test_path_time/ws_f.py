# -*- coding: utf-8 -*-
"""
simulate_full_frontend_batch.py （批量版本 + 日志写入）
--------------------------------------------------------
- 批量遍历 exp/exp3 下的所有子文件夹（如 Follow-up Questions）
- 对每个 .wav 调用 simulate_full_frontend() 发送音频并保存输出
- 输出路径与输入相同，只是文件名加 "_output.wav"
- 所有日志会一边打印到控制台，一边写入 test_path_time/10.txt
--------------------------------------------------------
"""
from tqdm.asyncio import tqdm_asyncio
from tqdm import tqdm
import asyncio
import json
import io
import time
import soundfile as sf
import numpy as np
import websockets
from pathlib import Path
WS_URL = "ws://127.0.0.1:18000/realtime"
BASE_DIR = Path("exp/exp3")
SAMPLE_RATE = 16000
CHUNK_SAMPLES = 256  # 16 ms per frame

class SpeakerSimulator:
    def __init__(self, total_duration: float, sr=SAMPLE_RATE, output_path: Path = None):
        self.sr = sr
        self.chunk_time = CHUNK_SAMPLES / sr
        self.total_samples = int(total_duration * sr)
        self.output_path = output_path
        self.audio_buffer = np.zeros(self.total_samples, dtype=np.float32)
        self.start_wall = None
        self.interrupted = False
        log(f"🧮 初始化输出 buffer，总时长 {total_duration:.2f}s ({self.total_samples} samples)")

    def reset_for_new_audio(self, wav_bytes: bytes, start_time: float):
        """在指定时间戳位置插入 TTS 音频"""
        data, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
        assert sr == self.sr, f"采样率不匹配: {sr}"

        start_sample = int(start_time * self.sr)
        available = self.total_samples - start_sample
        write_samples = min(len(data), max(0, available))

        if write_samples > 0:
            self.audio_buffer[start_sample:start_sample + write_samples] = data[:write_samples]
            log(f"🎵 在 {start_time:.2f}s 插入 TTS（{write_samples / self.sr:.2f}s）")

    def handle_interrupt(self):
        """收到打断后，后续保持静音"""
        self.interrupted = True
        log("🛑 播放被打断，后续输出静音")

    def save_output(self):
        """保存最终输出音频"""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(self.output_path, self.audio_buffer, self.sr)
        log(f"💾 已保存对齐音频: {self.output_path}, 时长 {len(self.audio_buffer) / self.sr:.2f}s")


async def mic_sender(ws, wav_path: Path):
    """实时发送音频帧"""
    data, sr = sf.read(str(wav_path), dtype="float32")
    if sr != SAMPLE_RATE:
        raise ValueError(f"采样率不匹配: {sr}")

    frame_time = CHUNK_SAMPLES / sr
    total_frames = int(np.ceil(len(data) / CHUNK_SAMPLES))
    log(f"🎙️ 开始发送 {wav_path.name}, 总帧 {total_frames}，时长 {len(data)/sr:.2f}s")
    t0 = time.perf_counter()

    for i in range(0, len(data), CHUNK_SAMPLES):
        chunk = data[i:i + CHUNK_SAMPLES]
        if len(chunk) < CHUNK_SAMPLES:
            chunk = np.pad(chunk, (0, CHUNK_SAMPLES - len(chunk)))
        await ws.send(chunk.tobytes())
        await asyncio.sleep(frame_time)

    await ws.send(json.dumps({"event": "end"}))
    log(f"📤 音频发送完毕，用时 {time.perf_counter() - t0:.2f}s")


async def simulate_full_frontend(wav_path: Path):
    """单个文件的前端模拟"""
    output_path = wav_path.parent / f"{wav_path.stem}_output.wav"
    data, sr = sf.read(str(wav_path), dtype="float32")
    total_duration = len(data) / sr

    async with websockets.connect(WS_URL, max_size=None) as ws:
        log(f"✅ 已连接后端: {WS_URL}")
        speaker = SpeakerSimulator(total_duration, output_path=output_path)
        last_tts_timestamp = None

        send_task = asyncio.create_task(mic_sender(ws, wav_path))

        async def receiver():
            nonlocal last_tts_timestamp
            while True:
                try:
                    msg = await ws.recv()
                except websockets.exceptions.ConnectionClosed:
                    log("⚠️ WebSocket 已关闭，结束接收循环")
                    break

                if isinstance(msg, bytes):
                    if last_tts_timestamp is None:
                        log("⚠️ 收到音频但无 timestamp，跳过写入")
                        continue
                    start_time = last_tts_timestamp
                    speaker.reset_for_new_audio(msg, start_time)
                    last_tts_timestamp = None
                    continue

                try:
                    obj = json.loads(msg)
                    event = obj.get("event")

                    if event == "tts_done":
                        last_tts_timestamp = obj["data"].get("timestamp")
                        log(f"🕒 收到 tts_done, timestamp={last_tts_timestamp}s")

                    elif event == "stop_audio":
                        speaker.handle_interrupt()

                    else:
                        log(f"📨 其他消息: {obj}")

                except Exception:
                    log(f"📨 文本消息: {msg}")

                await asyncio.sleep(0)

        recv_task = asyncio.create_task(receiver())
        await send_task
        speaker.save_output()

# ========== 基本配置 ==========


base_dir = Path("exp/exp3")
log_file = Path("test_path_time/11071046_log.txt")
log_file.parent.mkdir(parents=True, exist_ok=True)

read_all_subdirs = True    # false 时只读取 target_subdir
max_files = 10             # none 表示全部

target_subdir = "Repetition Requests"

def log(msg):
    with log_file.open("a") as f:
        f.write(msg + "\n")
    print(msg)

async def main():
    log_file.write_text("")
    log("========== 前端模拟开始 ==========")

    if read_all_subdirs:
        dirs = [d for d in base_dir.iterdir() if d.is_dir()]
    else:
        dirs = [base_dir / target_subdir]

    for subdir in sorted(dirs):
        log(f"\n进入目录: {subdir.name}")
        all_wavs = list(subdir.glob("*.wav"))
        wav_files = [f for f in all_wavs if not f.name.endswith("_output.wav")]
        wav_files = sorted(wav_files)

        if max_files:
            wav_files = wav_files[:max_files]
        if not wav_files:
            log(f"{subdir.name} 下没有 wav 文件")
            continue

        for wav_path in tqdm(wav_files, desc=subdir.name, ncols=80):
            log(f"处理文件: {wav_path.name}")
            await simulate_full_frontend(wav_path)

    log("========== 全部处理完成 ==========")

if __name__ == "__main__":
    asyncio.run(main())
if __name__ == "__main__":
    asyncio.run(main())
