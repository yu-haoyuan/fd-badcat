import json, soundfile as sf, numpy as np
from pathlib import Path
from tqdm import tqdm

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

    medium_root = Path("exp") / exp_name / "medium" / out_lang
    save_root_base = Path("exp") / exp_name / "dev" / data_lang

    for category in category_dev:
        category_path = medium_root / category
        save_root = save_root_base / category
        for folder in tqdm(list(category_path.iterdir()), desc=category):
            process_folder(folder, save_root)

if __name__ == "__main__":
    main()
