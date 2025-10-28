import subprocess
from pathlib import Path

def run_get_timing():
    # 指定输入音频目录（例如 Follow-up_Questions）
    root_dir = Path("/home/sds/output/merge/Follow-upQuestions")
    
    # 调用 get_timing.py
    cmd = [
        "python3",
        "fd-badcat/evaluation/interruption/get_timing.py",
        "--root_dir", str(root_dir)
    ]
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    run_get_timing()
