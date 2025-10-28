import json, soundfile as sf, numpy as np, shutil
from pathlib import Path
from tqdm import tqdm
from silero_vad import load_silero_vad, VADIterator
from module import asr, llm_qwen3o, tts, get_wav

sample_rate = 16000
window_size = 256
frame_sec = window_size / sample_rate
end_hold_sec = 0.64
end_hold_frames = int(end_hold_sec / frame_sec)

vad_model = load_silero_vad()
vad_iterator = VADIterator(vad_model, sampling_rate=sample_rate)

def stream_audio(path):
    import wave
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
    import wave
    audio_i16 = (np.clip(audio_float32, -1.0, 1.0) * 32768).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_i16.tobytes())

def stream_vad_asr_pipeline(audio_path, output_dir):
    basename = Path(audio_path).stem
    if not basename.startswith("clean_"):
        return
    jsonl_path = output_dir / f"{basename}_r.jsonl"
    if jsonl_path.exists():
        jsonl_path.unlink()
    state = "LISTEN"
    in_speech = False
    buffer = []
    silence_counter = 0
    frame_idx = 0
    turn_idx = 0
    media_time = 0.0
    import time, torch, numpy as np, soundfile as sf
    for frame in stream_audio(audio_path):
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
    vad_iterator.reset_states()

def build_clean_outputs(clean_root, merge_root):
    for category_dir in clean_root.iterdir():
        if not category_dir.is_dir():
            continue
        merge_dir = merge_root / category_dir.name
        merge_dir.mkdir(parents=True, exist_ok=True)
        for jsonl_file in category_dir.glob("clean_*.jsonl"):
            base_name = jsonl_file.stem.replace("_r", "")
            with open(jsonl_file, "r", encoding="utf-8") as f:
                lines = [l.strip() for l in f if l.strip()]
            if not lines:
                continue
            data = json.loads(lines[0])
            sys_start = data.get("sys_start", 0.0)
            wav_r1 = category_dir / f"{base_name}_r1.wav"
            if not wav_r1.exists():
                continue
            audio_data, sr = sf.read(wav_r1)
            silence_len = int(sys_start * sr)
            silence = np.zeros(silence_len, dtype=audio_data.dtype)
            full_audio = np.concatenate([silence, audio_data])
            out_path = category_dir / f"{base_name}_output.wav"
            sf.write(out_path, full_audio, sr)
            shutil.copy(out_path, merge_dir / out_path.name)

def main():
    exp_name = "exp1"
    data_lang = "dev_zh"
    category_dev = ["Follow-up Questions"]
    clean_root = Path("exp") / exp_name / "clean" / data_lang
    merge_root = Path("exp") / exp_name / "merge" / data_lang
    data_root = Path("exp") / exp_name / "dev" / data_lang
    for category in category_dev:
        wav_path = data_root / category
        output_root = clean_root / category
        wav_files = [w for w in get_wav(wav_path, "clean") if w.startswith("clean_")]
        for wav in tqdm(wav_files, desc=category):
            wav_file = wav_path / wav
            output_root.mkdir(parents=True, exist_ok=True)
            stream_vad_asr_pipeline(str(wav_file), output_root)
    build_clean_outputs(clean_root, merge_root)

if __name__ == "__main__":
    main()
