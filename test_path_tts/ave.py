import json
import argparse
from pathlib import Path
from statistics import mean

def safe_read_json(path: Path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️ 无法读取 {path}: {e}")
        return None


def get_all_json_files(exp, langs, categories):
    """获取所有语言中对应类别的_all.json文件内容"""
    data = []
    for lang in langs:
        for cat in categories:
            file_path = Path(f"exp/{exp}/score/{lang}/{cat}/{cat}_all.json")
            if file_path.exists():
                content = safe_read_json(file_path)
                if content:
                    data.append(content)
            else:
                print(f"❌ 未找到: {file_path}")
    return data


def ave_interrupt(exp, langs):
    """计算 interrupt 类别的平均值（中英共10个），包含所有样本（包括0）"""
    interrupt_cata = [
        "Follow-up Questions",
        "Negation or Dissatisfaction",
        "Repetition Requests",
        "Silence or Termination",
        "Topic Switching",
    ]
    all_data = get_all_json_files(exp, langs, interrupt_cata)
    if not all_data:
        return {}

    avg_resp = mean(d["average_RESPOND_score"] for d in all_data)
    avg_stop = mean(d["avg_latency_stop"] for d in all_data)
    avg_resp_lat = mean(d["avg_latency_resp"] for d in all_data)
    ftd_values = [d.get("avg_first_time_delay", 0) for d in all_data]
    avg_ftd = round(mean(ftd_values), 3)

    return {
        "Interruption Total Score": round(avg_resp * 100, 2),  # ✅ 改为百分数
        "avg_latency_stop": round(avg_stop, 3),
        "avg_latency_resp": round(avg_resp_lat, 3),
        "avg_first_time_delay": avg_ftd
    }


def ave_reject_resume(exp, langs):
    """计算 reject_resume_cata 的平均 RESUME 分数（包含0值）"""
    reject_resume_cata = [
        "Speech Directed at Others",
        "Third-party Speech_after",
        "User Real-time Backchannels",
    ]
    result = {}
    for cat in reject_resume_cata:
        files = get_all_json_files(exp, langs, [cat])
        if not files:
            continue
        avg_resume = mean(f.get("average_RESUME_score", 0) for f in files)
        ftd_values = [f.get("avg_first_time_delay", 0) for f in files]
        avg_ftd = round(mean(ftd_values), 3)
        result[cat] = {
            "average_RESUME_score": round(avg_resume, 3),
            "avg_first_time_delay": avg_ftd,
        }
    return result


def ave_reject_rate(exp, langs):
    """计算 reject_rate_cata 的平均 rejection rate（包含0值）"""
    reject_rate_cata = ["Pause Handling", "Third-party Speech_before"]
    result = {}
    for cat in reject_rate_cata:
        files = get_all_json_files(exp, langs, [cat])
        if not files:
            continue
        avg_reject = mean(f.get("reject_rate", 0) for f in files)
        ftd_values = [f.get("avg_first_time_delay", 0) for f in files]
        avg_ftd = round(mean(ftd_values), 3)
        result[cat] = {
            "reject_rate": round(avg_reject, 3),
            "avg_first_time_delay": avg_ftd,
        }
    return result


def compute_rejection_total_score(reject_resume_stats: dict, reject_rate_stats: dict) -> float:
    """
    Rejection Total Score（修改版）：
      - Third-party Speech_before / after 先取平均
      - 然后与其他三个类别（Speech Directed at Others, User Real-time Backchannels, Pause Handling）一起求平均
      - 总分除以4后乘100
    """
    # 各部分提取
    sda = reject_resume_stats.get("Speech Directed at Others", {}).get("average_RESUME_score", 0.0)
    urbc = reject_resume_stats.get("User Real-time Backchannels", {}).get("average_RESUME_score", 0.0)
    ph = reject_rate_stats.get("Pause Handling", {}).get("reject_rate", 0.0)

    tp_before = reject_rate_stats.get("Third-party Speech_before", {}).get("reject_rate", 0.0)
    tp_after = reject_resume_stats.get("Third-party Speech_after", {}).get("average_RESUME_score", 0.0)

    # 先合并 before/after
    third_party_avg = (tp_before + tp_after) / 2

    # 再取整体平均（分母=4）
    vals = [sda, urbc, ph, third_party_avg]
    return round(mean(vals) * 100, 3)


def compute_final(exp):
    langs = ["cn", "en"]
    print(f"🧮 开始计算 {exp} 汇总结果...")

    # ===== 各部分计算 =====
    interrupt_stats = ave_interrupt(exp, langs)
    reject_resume_stats = ave_reject_resume(exp, langs)
    reject_rate_stats = ave_reject_rate(exp, langs)
    reject_stats = {**reject_resume_stats, **reject_rate_stats}

    # ===== 全局 First Response Delay（所有样本，包括0）=====
    all_categories = (
        ["Follow-up Questions", "Negation or Dissatisfaction", "Repetition Requests",
         "Silence or Termination", "Topic Switching"] +
        ["Speech Directed at Others", "Third-party Speech_after", "User Real-time Backchannels",
         "Pause Handling", "Third-party Speech_before"]
    )
    all_files = get_all_json_files(exp, langs, all_categories)
    ftd_values = [f.get("avg_first_time_delay", 0) for f in all_files]
    avg_global_ftd = round(mean(ftd_values), 3) if ftd_values else 0.0

    # ===== 拒识总分（包含0值，百分制）=====
    rejection_total_score = compute_rejection_total_score(reject_resume_stats, reject_rate_stats)

    # ===== Total Delay（三项平均）=====
    delay_values = [
        avg_global_ftd,
        interrupt_stats["avg_latency_stop"],
        interrupt_stats["avg_latency_resp"]
    ]
    total_delay = round(mean(delay_values), 3)

    # ===== 最终结构 =====
    final_data = {
        "interrupt": interrupt_stats,
        "reject": reject_stats,
        "First Response Delay": avg_global_ftd,
        "Interruption Total Score": interrupt_stats.get("Interruption Total Score", 0.0),
        "Rejection Total Score": rejection_total_score,
        "Total Delay": total_delay
    }

    out_path = Path(f"exp/{exp}/score/all.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final_data, f, ensure_ascii=False, indent=2)
    print(f"✅ 汇总已保存到: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="统计各类别 all.json 的平均值并汇总")
    parser.add_argument("--exp", type=str, required=True, help="实验名称，例如 exp4")
    args = parser.parse_args()
    compute_final(args.exp)
