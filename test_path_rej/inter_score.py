import subprocess
from pathlib import Path
import sys
def run_get_timing(get_timing_dir):
    # 调用 get_timing.py
    cmd = [
        "python3",
        "evaluation/interruption/get_timing.py",
        "--root_dir", str(get_timing_dir)
    ]
    subprocess.run(cmd, check=True)

def run_get_trans(get_trans_dir: Path, lang: str):
    '''
    输入4个wav
    输出4个json
    '''
    cmd = [
        sys.executable, 
        f"evaluation/get_transcript/infer_{lang}.py",
        "--root_dir", str(get_trans_dir)
    ]
    subprocess.run(cmd, check=True)

def run_get_eval(get_eval_dir: Path):
    '''
    输入4个json
    输出./dev/json_group,由于这里会时刻刷新,暂时不分实验部署,而是从这里获取保存在exp_n下面
    '''
    cmd = [
        sys.executable,  # 使用当前环境的 python
        "evaluation/interruption/eval.py",
        "--data_dir", str(get_eval_dir)
    ]
    subprocess.run(cmd, check=True)

from evaluation.interruption.compute_first_response_delay import process_folder
import json
def run_get_ftd(get_ftd_dir: Path, save_dir: Path):
    ''' 
    first time delay计算首帧延迟
    输入两个json
    输出一个print
    '''
    avg, len_diffs = process_folder(str(get_ftd_dir))
    save_path = save_dir / "Follow-up Questions_ftd.json"
    result = {
    "folder": str(get_ftd_dir),
    "avg_first_time_delay": avg,
    "num_samples": len_diffs
    }
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"First time delay: {result}")

def main():
    exp = "exp1"
    lang = "dev_zh"
    categories = ["Follow-up Questions"]
    for category in categories:
        time_dir = Path(f"exp/{exp}/dev/{lang}/{category}")
        trans_dir = Path(f"exp/{exp}/dev/{lang}/{category}")
        ftd_dir = Path(f"exp/{exp}/score")
        
        # run_get_timing(time_dir)
        # run_get_trans(trans_dir, "cn")

        # run_get_eval(trans_dir)
        run_get_ftd(trans_dir, ftd_dir)

if __name__ == "__main__":
    main()