import subprocess
from pathlib import Path

def run_get_timing(get_timing_dir):
    # 调用 get_timing.py
    cmd = [
        "python3",
        "evaluation/interruption/get_timing.py",
        "--root_dir", str(get_timing_dir)
    ]
    subprocess.run(cmd, check=True)

def run_get_trans(get_eval_dir: Path, lang: str):
    cmd = [
        sys.executable, 
        f"evaluation/get_transcript/infer_{lang}.py",
        "--data_dir", str(get_eval_dir)
    ]

def run_get_eval(get_eval_dir: Path):
    cmd = [
        sys.executable,  # 使用当前环境的 python
        "evaluation/interruption/eval.py",
        "--data_dir", str(get_eval_dir)
    ]
    subprocess.run(cmd, check=True, cwd=str(project_root))

def main():
    exp = "exp1"
    lang = "dev_zh"
    categories = ["Follow-up Questions"]
    for category in categories:
        time_dir = Path(f"exp/{exp}/dev/{lang}/{category}")
        trans_dir = Path(f"exp/{exp}/dev/{lang}/{category}")
        run_get_timing(time_dir)
        run_get_trans(trans_dir, "zh")

if __name__ == "__main__":
    main()