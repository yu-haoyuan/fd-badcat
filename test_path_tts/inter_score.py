import subprocess
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
# 关于首帧延迟，interrupt和reject的Third-party_Speech_after,talk_to_others,Pause_Handling
# 一样，可以复用Third-party_Speech_before User_Real-time_Backchannels这两个需要单独拉出来

#------------interrupt三步走------------------------
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
    '''
    cmd = [
        sys.executable,  # 使用当前环境的 python
        "evaluation/interruption/eval.py",
        "--data_dir", str(get_eval_dir),
        "--output_dir", str(eval_output_dir)
    ]
    subprocess.run(cmd, check=True)
#------------interrupt三步走----------------------------------------------------------------------


#---------different-first-time-delay-------------------------------------------------------------
from evaluation.interruption.compute_first_response_delay import process_folder1
import json
def run_get_ftd(get_ftd_dir: Path, save_dir: Path):
    ''' 
    first time delay计算首帧延迟
    输入两个json
    输出一个print
    '''
    avg, len_diffs = process_folder1(str(get_ftd_dir))
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

import importlib.util, sys, pathlib

path = pathlib.Path("evaluation/rejection/Third-party_Speech/Third-party_Speech_before/compute_first_response_delay_before.py")
spec = importlib.util.spec_from_file_location("compute_first_response_delay_before", path)
module = importlib.util.module_from_spec(spec)
sys.modules["compute_first_response_delay_before"] = module
spec.loader.exec_module(module)

process_folder2 = module.process_folder2
def run_get_ftd_before(get_ftd_dir: Path, save_dir: Path):
    ''' 
    first time delay计算首帧延迟
    输入两个json
    输出一个print
    '''
    avg, len_diffs = process_folder2(str(get_ftd_dir))
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

path = pathlib.Path("evaluation/rejection/User_Real-time_Backchannels/compute_first_response_delay.py")
spec = importlib.util.spec_from_file_location("compute_first_response_delay", path)
module = importlib.util.module_from_spec(spec)
sys.modules["compute_first_response_delay"] = module
spec.loader.exec_module(module)

# 提取函数引用
process_folder3 = module.process_folder3
def run_get_ftd_bc(get_ftd_dir: Path, save_dir: Path):
    ''' 
    first time delay计算首帧延迟
    输入两个json
    输出一个print
    '''
    avg, len_diffs = process_folder3(str(get_ftd_dir))
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
#---------different-first-time-delay-------------------------------------------------------------


#-----------------reject-的三个特殊脚本----------------------
def run_get_ph_rr(get_eval_dir: Path, eval_output_dir: Path):
    '''
    返回reject_rate到score/reject_rate.json
    输入一个包含 *_sentence.json 和 *_output.json 的目录
    输出一个 json 文件 reject_rate.json 到指定 output_dir
    '''
    import subprocess, sys
    cmd = [
        sys.executable,
        "evaluation/rejection/Pause_Handling/compute_rejection_rate.py",
        "--data_dir", str(get_eval_dir),
        "--output_dir", str(eval_output_dir)
    ]
    subprocess.run(cmd, check=True)

def run_get_tpsb_rr(get_eval_dir: Path, eval_output_dir: Path):
    import subprocess, sys
    print("before-rr")
    cmd = [
        sys.executable,
        "evaluation/rejection/Third-party_Speech/Third-party_Speech_before/compute_rejection_rate_before.py",
        "--data_dir", str(get_eval_dir),
        "--output_dir", str(eval_output_dir)
    ]
    subprocess.run(cmd, check=True)

def run_get_sdao_eval(get_eval_dir: Path, eval_output_dir: Path):
    import subprocess, sys
    cmd = [
        sys.executable,
        "evaluation/rejection/Speech_Directe_at_Others/second_step/prepare_for_eval_first.py",
        "--data_dir", str(get_eval_dir),
        "--output_dir", str(eval_output_dir)
    ]
    subprocess.run(cmd, check=True)
#-----------------reject-的三个特殊脚本----------------------


def main():
    #-------需要改的参数----------
    exp = "exp4"
    lang = "dev_zh" #dev_en
    json_lang = "cn" #"en"
    
    #---------------------------------------------
    # categories = ["Follow-up Questions",
    #               "Negation or Dissatisfaction",
    #               "Repetition Requests",
    #               "Silence or Termination",
    #               "Topic Switching",

    #               "Pause_Handling", 
    #               "Speech_Directe_at_Others", 
    #               "Third-party Speech_before", 
    #               "User_Real-time_Backchannels"]
    # categories = ["Pause Handling"]
    root = Path(f"exp/{exp}/{lang}")
    score_root = Path(f"exp/{exp}/score")
    existing = {d.name for d in score_root.iterdir() if d.is_dir()}
    complete = []
    for d in root.iterdir():
        if not d.is_dir(): continue
        ins = [p.stem for p in d.glob("*.wav") if not p.name.endswith("_output.wav")]
        outs = {p.stem.removesuffix("_output") for p in d.glob("*_output.wav")}
        if ins and all(s in outs for s in ins): complete.append(d.name)

    print(complete)
    filtered = [name for name in complete if name not in existing]

 
    # filtered = ["Third-party Speech_before"]
    print(filtered)   

    #-------需要改的参数----------

    for category in filtered:
    #---------特殊-reject-----------------------------------
        if category == "Pause Handling":
            trans_dir_rej_ph = Path(f"exp/{exp}/{lang}/{category}")
            rej_ph_rr_dir = Path(f"exp/{exp}/score/{category}")

            run_get_trans(trans_dir_rej_ph, json_lang) #复用evaluation/get_transcript/infer_cn_en.py
            run_get_ph_rr(trans_dir_rej_ph, rej_ph_rr_dir) #特有的，需要得到详细信息
            run_get_ftd(trans_dir_rej_ph, rej_ph_rr_dir) #ftd可复用interrupt


        elif category == "Third-party Speech_before":
            print(f"✅")
            trans_dir_rej_tpsb = Path(f"exp/{exp}/{lang}/{category}")
            rej_tpsb_rr_dir = Path(f"exp/{exp}/score/{category}")

            #run_get_trans(trans_dir_rej_tpsb, json_lang)  #复用evaluation/get_transcript/infer_cn_en.py
            run_get_tpsb_rr(trans_dir_rej_tpsb, rej_tpsb_rr_dir) # 调用evaluation/rejection/Third-party_Speech/Third-party_Speech_before/compute_rejection_rate_before.py
            run_get_ftd_before(trans_dir_rej_tpsb, rej_tpsb_rr_dir) #单独的ftd

    #---------特殊----reject-----------------------------
            

    #---------------resume-----------------------------
        elif category == "Speech_Directe at Others":
            trans_dir_rej_sdao = Path(f"exp/{exp}/{lang}/{category}")
            trans_dir_rej_sdao_eval = Path(f"exp/{exp}/{lang}/{category}_eval")

            rej_sdao_rr_dir = Path(f"exp/{exp}/score/{category}")
            run_get_trans(trans_dir_rej_sdao, json_lang) #复用evaluation/get_transcript/infer_cn_en.py
            run_get_sdao_eval(trans_dir_rej_sdao, trans_dir_rej_sdao_eval) #执行prepare_for_eval_first.py，复制到Speech_Directe at Others_eval下面
            
            run_get_trans(trans_dir_rej_sdao_eval, json_lang) #继续复用evaluation/get_transcript/infer_cn_en.py
            run_get_eval(rej_sdao_rr_dir, rej_sdao_rr_dir) #继续复用evaluation/eval.py
            run_get_ftd(trans_dir_rej_sdao, rej_sdao_rr_dir) #ftd可复用interrupt


        elif category == "Third-party Speech after":
            trans_dir_rej_tpsa = Path(f"exp/{exp}/{lang}/{category}")
            rej_tpsa_rr_dir = Path(f"exp/{exp}/score/{category}")

            run_get_trans(trans_dir_rej_tpsa, json_lang) #复用evaluation/get_transcript/infer_cn_en.py
            run_get_eval(trans_dir_rej_tpsa, rej_tpsa_rr_dir) #继续复用evaluation/eval.py
            run_get_ftd(trans_dir_rej_tpsa, rej_tpsa_rr_dir) #ftd可复用interrupt

        elif category == "User Real-time Backchannels":
            trans_dir_rej_bc = Path(f"exp/{exp}/{lang}/{category}")
            rej_bc_rr_dir = Path(f"exp/{exp}/score/{category}")
            Path(rej_bc_rr_dir).mkdir(parents=True, exist_ok=True)

            run_get_trans(trans_dir_rej_bc, json_lang) #复用evaluation/get_transcript/infer_cn_en.py
            run_get_eval(trans_dir_rej_bc, rej_bc_rr_dir) #继续复用evaluation/eval.py
            run_get_ftd_bc(trans_dir_rej_bc, rej_bc_rr_dir) #单独的ftd
    #---------------resume-----------------------------


    #----------interrupt-----------------------------------
        else:
            #inter delay score
            time_file_dir = Path(f"exp/{exp}/{lang}/{category}")
            time_out_dir = Path(f"exp/{exp}/score/{category}")
            run_get_timing(time_file_dir, time_out_dir)

            #inter transcript
            trans_dir_int = Path(f"exp/{exp}/{lang}/{category}")
            run_get_trans(trans_dir_int, json_lang)

            #inter eval score
            run_get_eval(trans_dir_int, time_out_dir) #转录目录, 评分输出目录
            
            #first time delay
            run_get_ftd(trans_dir_int, time_out_dir) #转录目录, 评分输出目录 interrupt场景四个可复用
    #----------interrupt-----------------------------------



if __name__ == "__main__":
    main()