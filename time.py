#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import contextlib
from datetime import timedelta
from pathlib import Path
import wave

# 可选回退：如果安装了 soundfile，可以处理更多编码格式
try:
    import soundfile as sf
    _HAVE_SF = True
except Exception:
    _HAVE_SF = False


def duration_via_wave(p: Path) -> float:
    """用 wave 读取 WAV 时长（秒）。"""
    with contextlib.closing(wave.open(str(p), "rb")) as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        if rate == 0:
            raise ValueError("Invalid sample rate: 0")
        return frames / float(rate)


def duration_via_soundfile(p: Path) -> float:
    """可选：使用 soundfile 读取非常规 WAV 编码。"""
    if not _HAVE_SF:
        raise RuntimeError("soundfile not available")
    with sf.SoundFile(str(p)) as f:
        if f.samplerate == 0:
            raise ValueError("Invalid sample rate: 0")
        return len(f) / float(f.samplerate)


def get_wav_duration_seconds(p: Path) -> float:
    """优先 wave，失败回退 soundfile。"""
    try:
        return duration_via_wave(p)
    except Exception:
        if _HAVE_SF:
            return duration_via_soundfile(p)
        raise


def calc_folder_wav_duration(folder: Path, verbose=False):
    """统计某个文件夹下所有 wav 时长."""
    wav_files = list(folder.rglob("*.wav"))
    total = 0.0
    errors = []

    for w in wav_files:
        try:
            total += get_wav_duration_seconds(w)
        except Exception as e:
            errors.append((w, str(e)))

    return total, wav_files, errors


def main():
    parser = argparse.ArgumentParser(description="统计 dev_en 与 dev_zh 中每个子文件夹的 WAV 时长")
    parser.add_argument(
        "--root",
        default="data/dev",
        help="根目录（默认 data/dev）"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="打印错误文件"
    )
    args = parser.parse_args()

    root = Path(args.root)

    dev_en = root / "dev_en"
    dev_zh = root / "dev_zh"

    if not dev_en.exists() or not dev_zh.exists():
        print("❌ data/dev/dev_en 或 data/dev/dev_zh 不存在，请检查 --root 参数")
        return

    all_parent_folders = []
    all_parent_folders += sorted([p for p in dev_en.iterdir() if p.is_dir()])
    all_parent_folders += sorted([p for p in dev_zh.iterdir() if p.is_dir()])

    print("====== 每个子文件夹的统计结果 ======\n")

    for folder in all_parent_folders:
        total_sec, files, errors = calc_folder_wav_duration(folder, verbose=args.verbose)

        td = timedelta(seconds=int(round(total_sec)))

        print(f"{folder} :")
        print(f"  WAV 数量：{len(files)}")
        print(f"  总时长（秒）：{total_sec:.2f}")
        print(f"  总时长（天 时:分:秒）：{td}\n")

        if errors and args.verbose:
            print("  以下文件读取失败：")
            for f, msg in errors:
                print(f"    - {f} | {msg}")
            print()


if __name__ == "__main__":
    main()
