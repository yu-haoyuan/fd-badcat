#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
✅ Silero + Sherpa 流式处理管线（Clean版）
✅ 输入：以 clean_ 开头的干净音频
✅ 输出：clean_0001_0003_output.wav 直接保存在 /home/sds/output/clean/<类别>/
"""

import os
import wave
import numpy as np
import torch
import time
import json
from silero_vad import load_silero_vad, VADIterator
from module import asr, llm_qwen3o, tts, get_wav
import soundfile as sf
from pathlib import Path
from tqdm import tqdm

# ===============================
# 参数
# ===============================
SAMPLE_RATE = 16000
WINDOW_SIZE = 256
FRAME_SEC = WINDOW_SIZE / SAMPLE_RATE
END_HOLD_SEC = 0.64
END_HOLD_FRAMES = int(END_HOLD_SEC / FRAME_SEC)

vad_model = load_silero_vad()
vad_iterator = VADIterator(vad_model, sampling_rate=SAMPLE_RATE)


# ===============================
# 工具函数
# ===============================
def stream_audio(path):
    with wave.open(path, "rb") as wf:
        while True:
            data = wf.readframes(WINDOW_SIZE)
            if not data:
                break
            chunk = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            if len(chunk) < WINDOW_SIZE:
                break
            yield chunk


def detect_vad_frame(chunk):
    if not hasattr(detect_vad_frame, "buf"):
        detect_vad_frame.buf = np.zeros(0, dtype=np.float32)
    detect_vad_frame.buf = np.concatenate([detect_vad_frame.buf, chunk])
    if len(detect_vad_frame.buf) >= 2 * WINDOW_SIZE:
        tensor = torch.from_numpy(detect_vad_frame.buf[: 2 * WINDOW_SIZE])
        event = vad_iterator(tensor, return_seconds=True)
        detect_vad_frame.buf = np.zeros(0, dtype=np.float32)
        return event
    return None


def save_audio(path, audio_float32):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    audio_i16 = (np.clip(audio_float32, -1.0, 1.0) * 32768).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_i16.tobytes())


# ===============================
# 主流程
# ===============================
def stream_vad_asr_pipeline(audio_path: str, output_dir: str):
    basename = os.path.splitext(os.path.basename(audio_path))[0]
    jsonl_path = os.path.join(output_dir, f"{basename}_r.jsonl")
    if os.path.exists(jsonl_path):
        os.remove(jsonl_path)

    state = "LISTEN"
    in_speech = False
    buffer = []
    silence_counter = 0
    frame_idx = 0
    turn_idx = 0
    media_time = 0.0

    global_start = time.perf_counter()
    wall_time = lambda: time.perf_counter() - global_start

    for frame in stream_audio(audio_path):
        event = detect_vad_frame(frame)
        media_time = frame_idx * FRAME_SEC
        frame_idx += 1

        # ============ LISTEN 状态 ============
        if state == "LISTEN":
            if event and "start" in event and not in_speech:
                in_speech = True
                buffer = [frame]
                silence_counter = 0
            elif in_speech:
                buffer.append(frame)
                if event and "end" in event:
                    silence_counter = 1
                elif silence_counter > 0 and event and "start" in event:
                    silence_counter = 0
                elif silence_counter > 0:
                    silence_counter += 1
                    if silence_counter >= END_HOLD_FRAMES:
                        silence_counter = 0
                        in_speech = False
                        full_audio = np.concatenate(buffer)
                        buffer.clear()

                        # 保存用户说话段
                        user_path = os.path.join(output_dir, f"{basename}_u{turn_idx+1}.wav")
                        save_audio(user_path, full_audio)

                        # ASR
                        asr_start = time.perf_counter()
                        text = asr(user_path)
                        asr_elapsed = time.perf_counter() - asr_start

                        # LLM
                        llm_start = time.perf_counter()
                        reply = llm_qwen3o(text)
                        llm_elapsed = time.perf_counter() - llm_start

                        # TTS
                        r_path = os.path.join(output_dir, f"{basename}_r{turn_idx+1}.wav")
                        tts_start = time.perf_counter()
                        tts(reply, r_path)
                        tts_elapsed = time.perf_counter() - tts_start

                        audio_data, sr = sf.read(r_path)
                        duration = len(audio_data) / sr
                        sys_start = media_time + asr_elapsed + llm_elapsed + tts_elapsed

                        info = {
                            "turn": turn_idx + 1,
                            "user_end": round(media_time, 2),
                            "asr_time": round(asr_elapsed, 2),
                            "llm_time": round(llm_elapsed, 2),
                            "tts_time": round(tts_elapsed, 2),
                            "tts_dur": round(duration, 2),
                            "sys_start": round(sys_start, 2),
                            "tts_file": os.path.basename(r_path)
                        }
                        with open(jsonl_path, "a", encoding="utf-8") as f:
                            f.write(json.dumps(info, ensure_ascii=False) + "\n")

                        turn_idx += 1

    vad_iterator.reset_states()


import shutil

def build_clean_outputs():
    clean_root = Path("/home/sds/output/clean")
    merge_root = Path("/home/sds/output/merge")

    for category_dir in clean_root.iterdir():
        if not category_dir.is_dir():
            continue

        print(f"🧩 处理类别: {category_dir.name}")
        merge_dir = merge_root / category_dir.name
        merge_dir.mkdir(parents=True, exist_ok=True)

        for jsonl_file in category_dir.glob("*.jsonl"):
            base_name = jsonl_file.stem.replace("_r", "")
            with open(jsonl_file, "r", encoding="utf-8") as f:
                lines = [l.strip() for l in f if l.strip()]
            if not lines:
                print(f"⚠️ 空 JSONL 文件: {jsonl_file}")
                continue

            # 取第一轮对话 (turn 1)
            data = json.loads(lines[0])
            sys_start = data.get("sys_start", 0.0)
            wav_r1 = category_dir / f"{base_name}_r1.wav"
            if not wav_r1.exists():
                print(f"❌ 找不到 {wav_r1}")
                continue

            # 读取 r1 音频
            audio_data, sr = sf.read(wav_r1)
            silence_len = int(sys_start * sr)
            silence = np.zeros(silence_len, dtype=audio_data.dtype)
            full_audio = np.concatenate([silence, audio_data])

            # 保存 output.wav
            out_path = category_dir / f"{base_name}_output.wav"
            sf.write(out_path, full_audio, sr)
            print(f"✅ 生成: {out_path.name} (静音 {sys_start}s)")

            # 复制到 merge 目录
            shutil.copy(out_path, merge_dir / out_path.name)
            print(f"📦 复制到: {merge_dir / out_path.name}")

    print("🎯 全部类别处理完成。")


# ===============================
# 主入口
# ===============================
def main():
    wav_path = Path("/home/sds/data/dev/dev_zh/Follow-upQuestions")
    output_root = Path("/home/sds/output/clean/Follow-upQuestions")  # ✅ 扁平输出路径

    wav_files = [w for w in get_wav(wav_path, 'clean') if w.startswith("clean_")]
    print(f"找到 {len(wav_files)} 个 clean wav 文件进行处理")

    # for wav in tqdm(wav_files):
    #     wav_file = wav_path / wav
    #     tqdm.write(f"处理: {wav_file}")

    #     output_root.mkdir(parents=True, exist_ok=True)
    #     stream_vad_asr_pipeline(str(wav_file), str(output_root))

    #     # 检查输出完整性
    #     json_files_in_dir = list(Path(output_root).glob(f"{wav_file.stem}_r.jsonl"))
    #     if not json_files_in_dir:
    #         raise RuntimeError(f"未生成对应的 jsonl 文件: {wav_file.stem}")
    build_clean_outputs()

if __name__ == "__main__":
    main()
