import subprocess
from pathlib import Path
import sys
import argparse
import yaml
import os
def find_project_root(marker_name="fd-badcat"):
    """从当前文件向上查找名字为 marker_name 的目录，并返回其 Path。"""
    current = Path(__file__).resolve()
    for parent in current.parents:
        if parent.name == marker_name:
            return parent
    raise RuntimeError(f"未找到项目根目录 '{marker_name}'")

# 自动找到根目录 fd-badcat
ROOT = find_project_root("fd-badcat")

# 加入 sys.path 最前
sys.path.insert(0, str(ROOT))
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
    if lang == "cn":
        cmd = [
            sys.executable, 
            f"evaluation/get_transcript/infer_cn_en.py",
            "--root_dir", str(get_trans_dir)
        ]
        subprocess.run(cmd, check=True)
    else:
        cmd = [
            sys.executable, 
            f"evaluation/get_transcript/infer_en.py",
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
module2 = importlib.util.module_from_spec(spec)
sys.modules["compute_first_response_delay_before"] = module2
spec.loader.exec_module(module2)

process_folder2 = module2.process_folder2
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
module3 = importlib.util.module_from_spec(spec)
sys.modules["compute_first_response_delay"] = module3
spec.loader.exec_module(module3)

# 提取函数引用
process_folder3 = module3.process_folder3
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

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="test_path_tts/config.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    client_cfg = cfg.get("client", {})
    exp = client_cfg.get("exp", {})

    score_root = Path(f"exp/{exp}/score")
    score_root.mkdir(parents=True, exist_ok=True)
    langs = ["en", "cn"]

    existing_by_lang = {}
    for lg in langs:
        lang_dir = score_root / lg
        if lang_dir.exists():
            existing_by_lang[lg] = {d.name for d in lang_dir.iterdir() if d.is_dir()}
        else:
            existing_by_lang[lg] = set()

    # 用 set 去重，避免中英文重复加入相同类名
    complete = set()
    for dev in ["dev_zh", "dev_en"]:
        root = Path(f"exp/{exp}/{dev}")
        if not root.exists():
            print(f"跳过不存在的目录: {root}")
            continue
        for d in root.iterdir():
            if not d.is_dir(): 
                continue
            ins = [p.stem for p in d.glob("*.wav") if not p.name.endswith("_output.wav")]
            outs = {p.stem.removesuffix("_output") for p in d.glob("*_output.wav")}
            if ins and all(s in outs for s in ins):
                complete.add(d.name)

    # 打印所有完整类（去重后）
    complete = sorted(list(complete))
    print("\n检测到完整类别:")
    print(complete)

    # 分语言分别打印未打分的类
    for lang in langs:
        dev_name = "dev_zh" if lang == "cn" else "dev_en"
        filtered = [name for name in complete if name not in existing_by_lang[lang]]
        print(f"\n[{lang}] 还未评分{len(filtered)}个类:")
        print(filtered)


    #-------需要改的参数----------
    for lang in langs:
        if lang == "cn":
            dev_name = "dev_zh"
        else:
            dev_name = "dev_en"

        for category in filtered:
        #---------特殊-reject-----------------------------------
            if category == "Pause Handling":
                trans_dir_rej_ph = Path(f"exp/{exp}/{dev_name}/{category}")
                rej_ph_rr_dir = Path(f"exp/{exp}/score/{lang}/{category}")
                rej_ph_rr_dir.mkdir(parents=True, exist_ok=True)

                run_get_trans(trans_dir_rej_ph, lang) #复用evaluation/get_transcript/infer_cn_en.py
                run_get_ph_rr(trans_dir_rej_ph, rej_ph_rr_dir) #特有的，需要得到详细信息
                run_get_ftd(trans_dir_rej_ph, rej_ph_rr_dir) #ftd可复用interrupt


            elif category == "Third-party Speech_before":
                trans_dir_rej_tpsb = Path(f"exp/{exp}/{dev_name}/{category}")
                rej_tpsb_rr_dir = Path(f"exp/{exp}/score/{lang}/{category}")
                rej_tpsb_rr_dir.mkdir(parents=True, exist_ok=True)

                run_get_trans(trans_dir_rej_tpsb, lang)  #复用evaluation/get_transcript/infer_cn_en.py
                run_get_tpsb_rr(trans_dir_rej_tpsb, rej_tpsb_rr_dir) # 调用evaluation/rejection/Third-party_Speech/Third-party_Speech_before/compute_rejection_rate_before.py
                run_get_ftd_before(trans_dir_rej_tpsb, rej_tpsb_rr_dir) #单独的ftd

        #---------特殊-reject-----------------------------
                
        #---------------resume-----------------------------
            elif category == "Speech_Directe at Others":
                trans_dir_rej_sdao = Path(f"exp/{exp}/{dev_name}/{category}")
                trans_dir_rej_sdao_eval = Path(f"exp/{exp}/{dev_name}/{category}_eval")
                rej_sdao_rr_dir = Path(f"exp/{exp}/score/{lang}/{category}")
                rej_sdao_rr_dir.mkdir(parents=True, exist_ok=True)

                run_get_trans(trans_dir_rej_sdao, lang) #复用evaluation/get_transcript/infer_cn_en.py
                run_get_sdao_eval(trans_dir_rej_sdao, trans_dir_rej_sdao_eval) #执行prepare_for_eval_first.py，复制到Speech_Directe at Others_eval下面
                
                run_get_trans(trans_dir_rej_sdao_eval, lang) #继续复用evaluation/get_transcript/infer_cn_en.py
                run_get_eval(trans_dir_rej_sdao_eval, rej_sdao_rr_dir) #继续复用evaluation/eval.py
                run_get_ftd(trans_dir_rej_sdao, rej_sdao_rr_dir) #ftd可复用interrupt


            elif category == "Third-party Speech after":
                trans_dir_rej_tpsa = Path(f"exp/{exp}/{dev_name}/{category}")
                rej_tpsa_rr_dir = Path(f"exp/{exp}/score/{lang}/{category}")
                rej_tpsa_rr_dir.mkdir(parents=True, exist_ok=True)

                run_get_trans(trans_dir_rej_tpsa, lang) #复用evaluation/get_transcript/infer_cn_en.py
                run_get_eval(trans_dir_rej_tpsa, rej_tpsa_rr_dir) #继续复用evaluation/eval.py
                run_get_ftd(trans_dir_rej_tpsa, rej_tpsa_rr_dir) #ftd可复用interrupt

            elif category == "User Real-time Backchannels":
                trans_dir_rej_bc = Path(f"exp/{exp}/{dev_name}/{category}")
                rej_bc_rr_dir = Path(f"exp/{exp}/score/{lang}/{category}")
                rej_bc_rr_dir.mkdir(parents=True, exist_ok=True)

                run_get_trans(trans_dir_rej_bc, lang) #复用evaluation/get_transcript/infer_cn_en.py
                run_get_eval(trans_dir_rej_bc, rej_bc_rr_dir) #继续复用evaluation/eval.py
                run_get_ftd_bc(trans_dir_rej_bc, rej_bc_rr_dir) #单独的ftd
        #---------------resume-----------------------------


        #----------interrupt-----------------------------------
            else:
                #inter delay score
                time_file_dir = Path(f"exp/{exp}/{dev_name}/{category}")
                time_out_dir = Path(f"exp/{exp}/score/{lang}/{category}")
                time_out_dir.mkdir(parents=True, exist_ok=True)

                run_get_timing(time_file_dir, time_out_dir)

                #inter transcript
                trans_dir_int = Path(f"exp/{exp}/{dev_name}/{category}")
                run_get_trans(trans_dir_int, lang)

                #inter eval score
                run_get_eval(trans_dir_int, time_out_dir) #转录目录, 评分输出目录
                
                #first time delay
                run_get_ftd(trans_dir_int, time_out_dir) #转录目录, 评分输出目录 interrupt场景四个可复用
        #----------interrupt-----------------------------------



if __name__ == "__main__":
    main()