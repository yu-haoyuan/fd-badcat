#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
✅ Silero + Sherpa 流式处理管线（修复版）
✅ 640ms 静音确认 & 打断检测
✅ 相对路径读取（方便在不同机器运行）
✅ 所有路径拼接均在 main() 完成
"""

import os
import wave
import numpy as np
import torch
import time
import json
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
from silero_vad import load_silero_vad, VADIterator
from module import asr, llm, tts, get_time_wav, llm_qwen3o

# ===============================
# 全局参数
# ===============================
SAMPLE_RATE = 16000
WINDOW_SIZE = 256
FRAME_SEC = WINDOW_SIZE / SAMPLE_RATE
INTERRUPT_LIMIT = int(1.5 / FRAME_SEC)
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
# 主处理流程
# ===============================
def stream_vad_asr_pipeline(audio_path: Path, output_dir: Path):
    basename = audio_path.stem
    jsonl_path = output_dir / f"{basename}_r.jsonl"
    if jsonl_path.exists():
        jsonl_path.unlink()

    state = "LISTEN"
    in_speech = False
    buffer = []
    interrupt_buf = []
    interrupt_count = 0
    silence_counter = 0
    frame_idx = 0
    turn_idx = 0
    media_time = 0.0
    current_speak_turn = None
    current_sys_range = (None, None)

    global_start = time.perf_counter()
    wall_time = lambda: time.perf_counter() - global_start

    for frame in stream_audio(str(audio_path)):
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

                        user_path = output_dir / f"{basename}_u{turn_idx+1}.wav"
                        save_audio(user_path, full_audio)

                        # 模型调用
                        asr_start = time.perf_counter()
                        text = asr(str(user_path))
                        asr_elapsed = time.perf_counter() - asr_start

                        llm_start = time.perf_counter()
                        reply = llm_qwen3o(text)
                        llm_elapsed = time.perf_counter() - llm_start

                        r_path = output_dir / f"{basename}_r{turn_idx+1}.wav"
                        tts_start = time.perf_counter()
                        tts(reply, str(r_path))
                        tts_elapsed = time.perf_counter() - tts_start

                        audio_data, sr = sf.read(r_path)
                        duration = len(audio_data) / sr
                        sys_start = media_time + asr_elapsed + llm_elapsed + tts_elapsed
                        sys_finish = sys_start + duration
                        current_sys_range = (sys_start, sys_finish)

                        info = {
                            "turn": turn_idx + 1,
                            "user_end": round(media_time, 2),
                            "asr_time": round(asr_elapsed, 2),
                            "llm_time": round(llm_elapsed, 2),
                            "tts_time": round(tts_elapsed, 2),
                            "tts_dur": round(duration, 2),
                            "sys_start": round(sys_start, 2),
                            "tts_file": r_path.name
                        }
                        with open(jsonl_path, "a", encoding="utf-8") as f:
                            f.write(json.dumps(info, ensure_ascii=False) + "\n")

                        turn_idx += 1
                        state = "SPEAK"
                        current_speak_turn = turn_idx
                        interrupt_buf.clear()
                        interrupt_count = 0

        # ============ SPEAK 状态 ============
        elif state == "SPEAK":
            if event and "start" in event:
                interrupt_buf = [frame]
                interrupt_count = 1
            elif interrupt_buf:
                interrupt_buf.append(frame)
                interrupt_count += 1
                if event and "end" in event:
                    interrupt_buf.clear()
                    interrupt_count = 0
                elif interrupt_count >= INTERRUPT_LIMIT:
                    interrupt_time = round(media_time, 2)
                    if jsonl_path.exists():
                        lines = [l.strip() for l in jsonl_path.read_text(encoding="utf-8").splitlines() if l.strip()]
                        if lines:
                            last_obj = json.loads(lines[-1])
                            if last_obj.get("turn") == current_speak_turn:
                                last_obj["interrupt_time"] = interrupt_time
                                lines[-1] = json.dumps(last_obj, ensure_ascii=False)
                                jsonl_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
                    buffer = interrupt_buf.copy()
                    interrupt_buf.clear()
                    interrupt_count = 0
                    state = "LISTEN"
                    in_speech = True
                    silence_counter = 0

    # 文件结束仍在说话中
    if in_speech and buffer:
        full_audio = np.concatenate(buffer)
        user_path = output_dir / f"{basename}_u{turn_idx+1}.wav"
        save_audio(user_path, full_audio)
        asr_start = time.perf_counter()
        text = asr(str(user_path))
        asr_elapsed = time.perf_counter() - asr_start
        llm_start = time.perf_counter()
        reply = llm_qwen3o(text)
        llm_elapsed = time.perf_counter() - llm_start
        r_path = output_dir / f"{basename}_r{turn_idx+1}.wav"
        tts_start = time.perf_counter()
        tts(reply, str(r_path))
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
            "tts_file": r_path.name
        }
        with open(jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(info, ensure_ascii=False) + "\n")

    vad_iterator.reset_states()


# ===============================
# 主入口
# ===============================
from tqdm import tqdm
from pathlib import Path

def main():
    exp_name = "exp1"
    data_lang = "dev_zh" #dev_en
    out_lang = "medium_zh" #medium_en
    category_dev = ["Follow-up Questions"]

    data_root = Path("exp") / exp_name / "dev" / data_lang
    output_root = Path("exp") / exp_name / "medium" / out_lang
    output_root.mkdir(parents=True, exist_ok=True)

    for category in category_dev:
        wav_path = data_root / category
        out_dir_root = output_root / category
        out_dir_root.mkdir(parents=True, exist_ok=True)

        wav_files = get_time_wav(wav_path, time)
        print(f"找到 {len(wav_files)} 个 .wav 文件进行处理 ({category})")

        for wav in tqdm(wav_files):
            wav_file = wav_path / wav
            tqdm.write(f"处理: {wav_file}")

            output_dir = out_dir_root / wav_file.stem
            output_dir.mkdir(parents=True, exist_ok=True)

            stream_vad_asr_pipeline(wav_file, output_dir)

            wav_files_in_dir = list(output_dir.glob("*.wav"))
            json_files_in_dir = list(output_dir.glob("*.jsonl"))
            if not (len(wav_files_in_dir) == 4 and len(json_files_in_dir) == 1):
                raise RuntimeError(
                    f"输出数量错误: {output_dir} 中找到 {len(wav_files_in_dir)} 个wav, "
                    f"{len(json_files_in_dir)} 个json"
                )

if __name__ == "__main__":
    main()
