import json, soundfile as sf, numpy as np, shutil
from pathlib import Path
from tqdm import tqdm

def process_folder(folder, save_root, user_root):
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

        # ✅ 新增：复制同名用户音频
        user_wav = user_root / f"{folder.name}.wav"
        if user_wav.exists():
            shutil.copy(user_wav, save_root / f"{folder.name}.wav")



def main():
    root = Path("/home/sds/output")
    save_root = root / "merge"
    user_root_base = Path("/home/sds/data/dev/dev_zh")  # ✅ 顶层用户音频目录

    # ✅ 遍历所有类别（如 Follow-upQuestions、Interruption、TaskCompletion）
    for category in root.iterdir():
        if not category.is_dir() or category.name == "merge":
            continue

        user_root = user_root_base / category.name  # 对应类别下的用户音频路径
        if not user_root.exists():
            print(f"⚠️ 用户数据目录不存在: {user_root}")
            continue

        for folder in tqdm(list(category.iterdir()), desc=category.name):
            if folder.is_dir():
                process_folder(folder, save_root / category.name, user_root)

if __name__ == "__main__":
    main()
