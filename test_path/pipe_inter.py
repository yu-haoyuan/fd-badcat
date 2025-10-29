"""
✅ Silero + Sherpa 流式处理管线（修复版）
✅ 640ms 静音确认 & 打断检测
✅ 自动遍历 /home/sds/data 中的 wav 文件
✅ 每轮对话写入 /home/sds/output/{basename}_r.jsonl
"""

import os, wave, json, time, torch, soundfile as sf, numpy as np
from pathlib import Path
from tqdm import tqdm
from silero_vad import load_silero_vad, VADIterator
from module import asr, llm_qwen3o, tts, get_time_wav

sample_rate = 16000
window_size = 256
frame_sec = window_size / sample_rate
interrupt_limit = int(1.5 / frame_sec)
end_hold_sec = 0.64
end_hold_frames = int(end_hold_sec / frame_sec)

vad_model = load_silero_vad()
vad_iterator = VADIterator(vad_model, sampling_rate=sample_rate)

def stream_audio(path):
    with wave.open(path, "rb") as wf:
        while True:
            data = wf.readframes(window_size)
            if not data:
                break
            chunk = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            if len(chunk) < window_size:
                break
            yield chunk

def detect_vad_frame(chunk):
    if not hasattr(detect_vad_frame, "buf"):
        detect_vad_frame.buf = np.zeros(0, dtype=np.float32)
    detect_vad_frame.buf = np.concatenate([detect_vad_frame.buf, chunk])
    if len(detect_vad_frame.buf) >= 2 * window_size:
        tensor = torch.from_numpy(detect_vad_frame.buf[: 2 * window_size])
        event = vad_iterator(tensor, return_seconds=True)
        detect_vad_frame.buf = np.zeros(0, dtype=np.float32)
        return event
    return None

def save_audio(path, audio_float32):
    path.parent.mkdir(parents=True, exist_ok=True)
    audio_i16 = (np.clip(audio_float32, -1.0, 1.0) * 32768).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_i16.tobytes())

def stream_vad_asr_pipeline(audio_path, output_dir):
    basename = Path(audio_path).stem
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
    for frame in stream_audio(str(audio_path)):
        event = detect_vad_frame(frame)
        media_time = frame_idx * frame_sec
        frame_idx += 1
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
                    if silence_counter >= end_hold_frames:
                        silence_counter = 0
                        in_speech = False
                        full_audio = np.concatenate(buffer)
                        buffer.clear()
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
                        turn_idx += 1
                        state = "SPEAK"
                        current_speak_turn = turn_idx
                        interrupt_buf.clear()
                        interrupt_count = 0
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
                elif interrupt_count >= interrupt_limit:
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
    vad_iterator.reset_states()

def process_folder(folder, save_root):
    jsonl = folder / f"{folder.name}_r.jsonl"
    if not jsonl.exists():
        return
    data = [json.loads(l) for l in open(jsonl) if l.strip()]
    data.sort(key=lambda x: x["sys_start"])
    segs, cur_len, sr = [], 0, 16000
    for s in data:
        wav, sr = sf.read(folder / s["tts_file"])
        if wav.ndim > 1:
            wav = wav[:, 0]
        pad = int(max(0.0, s["sys_start"] - cur_len) * sr)
        if pad > 0:
            segs.append(np.zeros(pad, dtype=wav.dtype))
            cur_len += pad / sr
        cut = None
        if "interrupt_time" in s and s["sys_start"] + s["tts_dur"] > s["interrupt_time"]:
            cut = int(max(0.0, s["interrupt_time"] - s["sys_start"]) * sr)
        if cut is not None:
            wav = wav[:cut]
        segs.append(wav)
        cur_len += len(wav) / sr
    if segs:
        save_root.mkdir(parents=True, exist_ok=True)
        out = save_root / f"{folder.name}_output.wav"
        sf.write(out, np.concatenate(segs), sr)
        print(out)

def main():
    exp_name = "exp1"
    data_lang = "dev_zh"
    out_lang = "medium_zh"
    category_dev = ["Follow-up Questions"]

    data_root = Path("exp") / exp_name / "dev" / data_lang
    output_root = Path("exp") / exp_name / "medium" / out_lang
    for category in category_dev:
        wav_path = data_root / category
        medium_path = output_root / category
        medium_path.mkdir(parents=True, exist_ok=True)
        wav_files = get_time_wav(wav_path)
        for wav in tqdm(wav_files, desc=category):
            wav_file = wav_path / wav
            output_dir = medium_path / wav_file.stem
            output_dir.mkdir(parents=True, exist_ok=True)
            stream_vad_asr_pipeline(wav_file, output_dir)
            wav_files_in_dir = list(output_dir.glob("*.wav"))
            json_files_in_dir = list(output_dir.glob("*.jsonl"))
            if not (len(wav_files_in_dir) == 4 and len(json_files_in_dir) == 1):
                raise RuntimeError(f"输出数量错误: {output_dir} 中找到 {len(wav_files_in_dir)} 个wav, {len(json_files_in_dir)} 个json")
            process_folder(output_dir, wav_path)

if __name__ == "__main__":
    main()
