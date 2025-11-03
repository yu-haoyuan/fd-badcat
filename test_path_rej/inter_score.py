import subprocess
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
def run_get_timing(get_timing_dir, time_out_dir):
    # 调用 get_timing.py
    cmd = [
        "python3",
        "evaluation/interruption/get_timing.py",
        "--root_dir", str(get_timing_dir),
        "--result_dir", str(time_out_dir)
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

def run_get_eval(get_eval_dir: Path, eval_output_dir: Path):
    '''
    输入4个json
    输出./dev/json_group,由于这里会时刻刷新,暂时不分实验部署,而是从这里获取保存在exp_n下面
    '''
    cmd = [
        sys.executable,  # 使用当前环境的 python
        "evaluation/interruption/eval.py",
        "--data_dir", str(get_eval_dir),
        "--output_dir", str(eval_output_dir)
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
    save_path = save_dir / f"{get_ftd_dir.stem}_ftd.json"
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
    #语言
    lang = "dev_zh" #dev_en
    json_lang = "cn" #"en"
    
    #Negation or Dissatisfaction
    #---------------------------------------------
    categories = ["Follow-up Questions",
                  "Negation or Dissatisfaction",
                  "Repetition Requests",
                  "Silence or Termination",
                  "Topic Switching",

                  "Pause_Handling", 
                  "Speech_Directe_at_Others", 
                  "Third-party_Speech", 
                  "User_Real-time_Backchannels"]
    for category in categories:
    #----------reject-----------------------------------
        if category == "Pause_Handling":
            #rej transcript
            trans_dir_rej = Path(f"exp/{exp}/dev/{lang}/{category}")
            run_get_trans(trans_dir_rej, json_lang)

            
        if category == "Speech_Directe_at_Others":
            #rej transcript

        if category == "Third-party_Speech":
            #rej transcript
            trans_dir_rej = Path(f"exp/{exp}/dev/{lang}/{category}")
            run_get_trans(trans_dir_rej, json_lang)



        if category == "User_Real-time_Backchannels":
            #rej transcript
            trans_dir_rej = Path(f"exp/{exp}/dev/{lang}/{category}")
            run_get_trans(trans_dir_rej, json_lang)

        else:    
    #----------interrupt-----------------------------------
            #inter delay score
            time_file_dir = Path(f"exp/{exp}/dev/{lang}/{category}")
            time_out_dir = Path(f"exp/{exp}/score/{category}")
            run_get_timing(time_file_dir, time_out_dir)

            #inter transcript
            trans_dir_int = Path(f"exp/{exp}/dev/{lang}/{category}")
            run_get_trans(trans_dir_int, json_lang)

            #inter eval score
            run_get_eval(trans_dir_int, time_out_dir)
            
            #first time delay
            run_get_ftd(trans_dir_int, time_out_dir)
        



if __name__ == "__main__":
    main()