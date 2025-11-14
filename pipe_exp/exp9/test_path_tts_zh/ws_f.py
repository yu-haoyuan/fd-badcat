# -*- coding: utf-8 -*-
"""
simulate_full_frontend_batch.py （批量版本 + 日志写入）
--------------------------------------------------------
- 批量遍历 exp/exp3 下的所有子文件夹（如 Follow-up Questions）
- 对每个 .wav 调用 simulate_full_frontend() 发送音频并保存输出
- 输出路径与输入相同，只是文件名加 "_output.wav"
- 所有日志会一边打印到控制台，一边写入 test_path_time/11071046_log_*.txt
--------------------------------------------------------
"""
import argparse
from pathlib import Path
from datetime import datetime
import asyncio
import json
import io
import time

import numpy as np
import soundfile as sf
import websockets
from tqdm import tqdm
import yaml
# ========== 基本配置 ==========
WS_URL = None
SAMPLE_RATE = 16000
CHUNK_SAMPLES = 256  # 16 ms per frame

# ---------------- 日志控制类 ----------------
LOG_MAX_SIZE = 5 * 1024 * 1024  # 5MB

class RotatingLogger:
    def __init__(self, base_path: Path, max_size: int = LOG_MAX_SIZE):
        self.base_path = base_path
        self.max_size = max_size
        self.file_index = 1
        self.current_file = self._get_log_path()
        self.current_file.write_text("")  # 初始化清空首个日志

    def _get_log_path(self) -> Path:
        """根据当前序号生成日志文件路径"""
        return self.base_path.with_name(f"{self.base_path.name}_{self.file_index}.txt")

    def _rotate_if_needed(self):
        """当文件超过阈值时自动切分"""
        if self.current_file.exists() and self.current_file.stat().st_size >= self.max_size:
            self.file_index += 1
            self.current_file = self._get_log_path()
            self.current_file.write_text("")
            print(f"\n日志切分: {self.current_file.name}")

    def log(self, msg: str):
        ts = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
        line = f"{ts} {msg}"
        self._rotate_if_needed()
        with self.current_file.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
        print(line)


# ---------------- 播放端模拟器 ----------------
class SpeakerSimulator:
    def __init__(self, total_duration: float, sr: int = SAMPLE_RATE, output_path: Path | None = None, *, log=print):
        self.log = log
        self.sr = sr
        self.chunk_time = CHUNK_SAMPLES / sr
        self.total_samples = int(total_duration * sr)
        self.output_path = output_path
        self.audio_buffer = np.zeros(self.total_samples, dtype=np.float32)
        self.start_wall = None
        self.interrupted = False
        self.log(f"初始化输出 buffer，总时长 {total_duration:.2f}s（{self.total_samples} samples）")

    def reset_for_new_audio(self, wav_bytes: bytes, start_time: float):
        """在指定时间戳位置插入 TTS 音频"""
        data, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
        assert sr == self.sr, f"采样率不匹配: {sr} != {self.sr}"

        start_sample = int(start_time * self.sr)
        available = self.total_samples - start_sample
        write_samples = min(len(data), max(0, available))

        if write_samples > 0:
            self.audio_buffer[start_sample:start_sample + write_samples] = data[:write_samples]
            self.log(f"在 {start_time:.2f}s 插入 TTS（{write_samples / self.sr:.2f}s）")
        else:
            self.log(f"丢弃超出长度的 TTS 片段，start={start_time:.3f}s")

    def handle_interrupt(self):
        """收到打断后，后续保持静音（这里只记录标记，具体静音逻辑按需扩展）"""
        self.interrupted = True
        self.log("播放被打断，记录中断标记")

    def save_output(self):
        """保存最终输出音频"""
        if self.output_path is None:
            self.log("未指定输出路径，跳过保存")
            return
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(self.output_path, self.audio_buffer, self.sr)
        self.log(f"已保存对齐音频: {self.output_path}，时长 {len(self.audio_buffer) / self.sr:.2f}s")


# ---------------- 发送端：实时推送麦克风帧 ----------------
async def mic_sender(ws, wav_path: Path, *, log=print):
    """以实时节奏发送音频帧到后端"""
    data, sr = sf.read(str(wav_path), dtype="float32")
    if sr != SAMPLE_RATE:
        raise ValueError(f"采样率不匹配: {sr} != {SAMPLE_RATE}")

    frame_time = CHUNK_SAMPLES / sr
    total_frames = int(np.ceil(len(data) / CHUNK_SAMPLES))
    log(f"开始发送 {wav_path.name}，总帧 {total_frames}，时长 {len(data)/sr:.2f}s")
    t0 = time.perf_counter()

    for i in range(0, len(data), CHUNK_SAMPLES):
        chunk = data[i:i + CHUNK_SAMPLES]
        if len(chunk) < CHUNK_SAMPLES:
            chunk = np.pad(chunk, (0, CHUNK_SAMPLES - len(chunk)))
        await ws.send(chunk.tobytes())
        await asyncio.sleep(frame_time)

    await ws.send(json.dumps({"event": "end"}))
    log(f"音频发送完毕，用时 {time.perf_counter() - t0:.2f}s")


# ---------------- 单文件前端模拟 ----------------
async def simulate_full_frontend(wav_path: Path, *, log=print, lang, exp):
    """
    单个文件的前端模拟
    依赖：
      - SpeakerSimulator(total_duration: float, sr: int, output_path: Path, log=...)
      - mic_sender(ws, wav_path, log=...)
      - 服务器通过 WebSocket 发送:
          * 二进制音频帧（bytes）
          * 文本 JSON 消息，包含 event 字段，如 tts_done/stop_audio/其他
    """
    output_path = wav_path.parent / f"{wav_path.stem}_output.wav"
    data, sr = sf.read(str(wav_path), dtype="float32")
    total_duration = len(data) / sr

    async with websockets.connect(WS_URL, max_size=None) as ws:
        log(f"已连接后端: {WS_URL}")
        await ws.send(json.dumps({
            "event": "config",
            "data": {"lang": lang,
                     "exp": exp }
        }))
        speaker = SpeakerSimulator(total_duration, sr=sr, output_path=output_path, log=log)
        last_tts_timestamp = None

        async def receiver():
            nonlocal last_tts_timestamp
            while True:
                try:
                    msg = await ws.recv()
                except websockets.exceptions.ConnectionClosed:
                    log("WebSocket 已关闭，结束接收循环")
                    break

                # 二进制音频数据
                if isinstance(msg, bytes):
                    if last_tts_timestamp is None:
                        log("收到音频但无 timestamp，跳过写入")
                        continue
                    speaker.reset_for_new_audio(msg, last_tts_timestamp)
                    last_tts_timestamp = None
                    continue

                # 文本消息
                try:
                    obj = json.loads(msg)
                except Exception:
                    log(f"文本消息（非 JSON）: {msg}")
                    continue

                event = obj.get("event")
                if event == "tts_done":
                    last_tts_timestamp = obj.get("data", {}).get("timestamp")
                    log(f"收到 tts_done, timestamp={last_tts_timestamp}s")
                elif event == "stop_audio":
                    speaker.handle_interrupt()
                else:
                    log(f"其他消息: {obj}")

                await asyncio.sleep(0)

        # 发送与接收并行
        send_task = asyncio.create_task(mic_sender(ws, wav_path, log=log))
        recv_task = asyncio.create_task(receiver())

        # 等待发送完成，再关闭连接并等待接收结束
        await send_task
        await ws.close()
        await recv_task

        speaker.save_output()
        log(f"已保存 TTS 输出: {output_path}")


# ---------------- 批量主逻辑 ----------------



async def main():
    global WS_URL

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="test_path_tts/config.yaml", help="YAML配置文件路径")
    args = parser.parse_args()

    # 读取 YAML 配置
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    client_cfg = cfg.get("client", {})

    lang = client_cfg.get("lang", {})
    exp = client_cfg.get("exp", {})
    port = client_cfg.get("port", {})

    WS_URL = f"ws://127.0.0.1:{port}/realtime"

    if lang == "zhen":
        langs = ["zh", "en"]
    else:
        langs = [lang]

    for cur_lang in langs:
        base_dir = Path(f"exp/{exp}/dev_{cur_lang}")
        log_base = Path(f"exp/{exp}/{exp}_lg_{cur_lang}_p")
        log_base.parent.mkdir(parents=True, exist_ok=True)

        logger = RotatingLogger(log_base)
        log = logger.log
        log(f"========== 前端模拟开始 ({cur_lang}) ==========")

        read_all_subdirs = False
        max_files = None
        target_subdir = ["Topic Switching", "User Real-time Backchannels"]

        if read_all_subdirs:
            dirs = [d for d in base_dir.iterdir() if d.is_dir()]
        else:
            dirs = [base_dir / subdir for subdir in target_subdir]

        for subdir in tqdm(sorted(dirs), desc=f"{cur_lang} 目录进度", ncols=80):
            log(f"\n进入目录: {subdir.name}")
            all_wavs = list(subdir.glob("*.wav"))
            wav_files = [f for f in all_wavs if not f.name.endswith("_output.wav")]
            wav_files = sorted(wav_files)

            if max_files:
                wav_files = wav_files[:max_files]
            if not wav_files:
                log(f"{subdir.name} 下没有可处理的 wav 文件")
                continue

            for wav_path in tqdm(wav_files, desc=subdir.name, ncols=80):
                output_path = wav_path.with_name(wav_path.stem + "_output.wav")

                if output_path.exists():
                    log(f"跳过已处理文件: {wav_path.name}")
                    continue

                log(f"开始处理文件: {wav_path.name}")
                try:
                    await simulate_full_frontend(wav_path, log=log, lang=cur_lang, exp=exp)
                    log(f"处理完成: {wav_path.name}")
                except Exception as e:
                    log(f"处理 {wav_path.name} 出错: {e}")

                log("-----------------------------------")

        log(f"========== 全部处理完成 ({cur_lang}) ==========")

# ---------------- 入口 ----------------
if __name__ == "__main__":
    asyncio.run(main())