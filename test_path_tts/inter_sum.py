import json
from pathlib import Path
from collections import Counter


def _stats_by_axis(records):
    """计算行为轴统计（仅用 'C' 轴），返回 ratios 里有 'C_RESPOND' 的占比。"""
    axes = {"C": Counter()}
    for rec in records:
        for tag in rec.get("behaviour", []):
            if tag and tag[0] == "C":
                axes["C"][tag] += 1
    totals = {ax: sum(cnt.values()) for ax, cnt in axes.items()}
    ratios = {
        ax: {tag: (cnt / totals[ax] if totals[ax] else 0.0) for tag, cnt in counter.items()}
        for ax, counter in axes.items()
    }
    fmt_ratios = {ax: {k: round(v, 2) for k, v in ratios[ax].items()} for ax in ["C"]}
    return fmt_ratios.get("C", {}).get("C_RESPOND", 0.0)

def _stats_by_axis_resume(records):
    """计算行为轴统计（仅用 'C' 轴），返回 ratios 里有 'resume' 的占比。"""
    axes = {"C": Counter()}
    for rec in records:
        for tag in rec.get("behaviour", []):
            if tag and tag[0] == "C":
                axes["C"][tag] += 1
    totals = {ax: sum(cnt.values()) for ax, cnt in axes.items()}
    ratios = {
        ax: {tag: (cnt / totals[ax] if totals[ax] else 0.0) for tag, cnt in counter.items()}
        for ax, counter in axes.items()
    }
    fmt_ratios = {ax: {k: round(v, 2) for k, v in ratios[ax].items()} for ax in ["C"]}
    return fmt_ratios.get("C", {}).get("C_RESUME", 0.0)


def _safe_read_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _read_behaviour_records(behaviour_path_base: Path) -> list:
    """兼容 {cat}_behaviour.jsonl 或 {cat}_behaviour.json"""
    jsonl = behaviour_path_base.with_suffix(".jsonl")
    js = behaviour_path_base.with_suffix(".json")
    records = []
    if jsonl.exists():
        with open(jsonl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return records
    if js.exists():
        data = _safe_read_json(js)
        return data if isinstance(data, list) else [data]
    raise FileNotFoundError(f"行为评估文件不存在: {jsonl} 或 {js}")


def build_single_category(exp: str, category: str):
    """
    读取 exp/{exp}/score/{category} 下的三个结果文件，
    生成同目录下 {category}_all.json
    """
    cat_dir = Path(f"exp/{exp}/score") / category

    latency_path = cat_dir / f"{category}_latency_results.json"
    ftd_path = cat_dir / f"{category}_ftd.json"
    behaviour_base = cat_dir / f"{category}_content_tags.json"

    # 1) latency
    latency_data = _safe_read_json(latency_path)
    avg_latency_stop = latency_data.get("average_latency", {}).get("avg_latency_stop")
    avg_latency_resp = latency_data.get("average_latency", {}).get("avg_latency_resp")

    # 2) ftd
    ftd_data = _safe_read_json(ftd_path)
    avg_first_time_delay = ftd_data.get("avg_first_time_delay")

    # 3) behaviour
    records = _read_behaviour_records(behaviour_base)
    average_RESPOND_score = _stats_by_axis(records)

    out_data = {
        "avg_latency_stop": avg_latency_stop,
        "avg_latency_resp": avg_latency_resp,
        "avg_first_time_delay": avg_first_time_delay,
        "average_RESPOND_score": average_RESPOND_score
    }

    out_path = cat_dir / f"{category}_all.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False, indent=2)
    print(f"✅ 已生成: {out_path}")

def reject_rate_category(exp: str, category: str):
    """
    处理 'Pause Handling' 和 'User Real-time Backchannels' 类别的特殊需求，
    计算 avg_first_time_delay 和 reject_rate。
    """
    cat_dir = Path(f"exp/{exp}/score") / category

    ftd_path = cat_dir / f"{category}_ftd.json"
    reject_rate_path = cat_dir / f"reject_rate.json"

    # 1) ftd
    ftd_data = _safe_read_json(ftd_path)
    avg_first_time_delay = ftd_data.get("avg_first_time_delay")

    # 2) reject_rate
    reject_rate_data = _safe_read_json(reject_rate_path)
    total = reject_rate_data.get("total", 0)
    ahead = reject_rate_data.get("ahead", 0)
    reject_rate = ahead / total if total else 0.0

    out_data = {
        "avg_first_time_delay": avg_first_time_delay,
        "reject_rate": reject_rate
    }

    out_path = cat_dir / f"{category}_all.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False, indent=2)
    print(f"✅ 已生成: {out_path}")


def reject_resume_category(exp: str, category: str):
    """
    处理 'Speech Directed at Others'，'Third-party Speech_after' 和 'User Real-time Backchannels' 类别，
    计算 avg_first_time_delay 和 RESUME 的占比。
    """
    cat_dir = Path(f"exp/{exp}/score") / category

    ftd_path = cat_dir / f"{category}_ftd.json"
    behaviour_base = cat_dir / f"{category}_content_tags.json"

    # 1) ftd
    ftd_data = _safe_read_json(ftd_path)
    avg_first_time_delay = ftd_data.get("avg_first_time_delay")

    # 2) behaviour and RESUME score
    records = _read_behaviour_records(behaviour_base)
    average_RESUME_score = _stats_by_axis_resume(records)

    out_data = {
        "avg_first_time_delay": avg_first_time_delay,
        "average_RESUME_score": average_RESUME_score
    }

    out_path = cat_dir / f"{category}_all.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False, indent=2)
    print(f"✅ 已生成: {out_path}")


def calculate_average_of_keys(exp: str, categories: list):
    """
    计算每个类别的 {category}_all.json 中相同键对应的值的平均数
    """
    all_data = []

    # 遍历每个类别的all.json文件
    for category in categories:
        cat_dir = Path(f"exp/{exp}/score") / category
        all_file_path = cat_dir / f"{category}_all.json"
        try:
            data = _safe_read_json(all_file_path)
            all_data.append(data)
        except Exception as e:
            print(f"⚠️ 处理 {category} 时出错: {e}")

    # 获取所有键的集合
    keys = all_data[0].keys()

    # 计算每个键的平均值
    averages = {}
    for key in keys:
        values = [data.get(key, 0) for data in all_data]
        averages[key] = sum(values) / len(values)

    # 打印每个键的平均值
    for key, avg in averages.items():
        print(f"Key: {key}, Average: {avg:.2f}")

def main():
    exp = "exp3"
    base_score_dir = Path(f"exp/{exp}/score")
    interrupt_cata = ["Follow-up Questions", 
                      "Negation or Dissatisfaction",
                      "Repetition Requests",
                      "Silence or Termination",
                      "Topic Switching"]
    # 自动遍历 score 下的所有文件夹
    for cat_dir in interrupt_cata:
        try:
            build_single_category(exp, cat_dir)
        except Exception as e:
            print(f"⚠️ 处理 {cat_dir} 时出错: {e}")

    calculate_average_of_keys(exp, interrupt_cata)
    
    reject_rate_cata = ["Pause Handling",
                        "Third-party Speech_before"]
    for rcat_dir in reject_rate_cata:
        try:
            reject_rate_category(exp, rcat_dir)
        except Exception as e:
            print(f"⚠️ 处理 {rcat_dir} 时出错: {e}")  


    reject_resume_cata = ["Speech Directed at Others",
                          "Third-party Speech_after",
                          "User Real-time Backchannels"]   
    for rcat_dir in reject_resume_cata:
        try:
            reject_resume_category(exp, rcat_dir)
        except Exception as e:
            print(f"⚠️ 处理 {rcat_dir} 时出错: {e}")  


if __name__ == "__main__":
    main()
