import json
import argparse
from pathlib import Path
from collections import Counter
import yaml

# ===============================
# 工具函数
# ===============================

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
    """计算行为轴统计（仅用 'C' 轴），返回 ratios 里有 'C_RESUME' 的占比。"""
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


# ===============================
# 主处理函数（增加 lang 参数）
# ===============================

def build_single_category(exp: str, category: str, lang: str):
    """读取 exp/{exp}/score/{lang}/{category} 下的结果文件，生成 {category}_all.json"""
    cat_dir = Path(f"exp/{exp}/score/{lang}") / category

    latency_path = cat_dir / f"{category}_latency_results.json"
    ftd_path = cat_dir / f"{category}_ftd.json"
    behaviour_base = cat_dir / f"{category}_content_tags.json"

    latency_data = _safe_read_json(latency_path)
    avg_latency_stop = latency_data.get("average_latency", {}).get("avg_latency_stop")
    avg_latency_resp = latency_data.get("average_latency", {}).get("avg_latency_resp")

    ftd_data = _safe_read_json(ftd_path)
    avg_first_time_delay = ftd_data.get("avg_first_time_delay")

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


def reject_rate_category(exp: str, category: str, lang: str):
    """处理 'Pause Handling' 与 'Third-party Speech_before' 类别。"""
    cat_dir = Path(f"exp/{exp}/score/{lang}") / category

    ftd_path = cat_dir / f"{category}_ftd.json"
    reject_rate_path = cat_dir / f"reject_rate.json"

    ftd_data = _safe_read_json(ftd_path)
    avg_first_time_delay = ftd_data.get("avg_first_time_delay")

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


def reject_resume_category(exp: str, category: str, lang: str):
    """处理 'Speech Directed at Others'、'Third-party Speech_after'、'User Real-time Backchannels' 类别。"""
    cat_dir = Path(f"exp/{exp}/score/{lang}") / category

    ftd_path = cat_dir / f"{category}_ftd.json"
    behaviour_base = cat_dir / f"{category}_content_tags.json"

    ftd_data = _safe_read_json(ftd_path)
    avg_first_time_delay = ftd_data.get("avg_first_time_delay")

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


def calculate_average_of_keys(exp: str, categories: list, lang: str):
    """计算每个类别 {category}_all.json 中相同键的平均值"""
    all_data = []
    for category in categories:
        cat_dir = Path(f"exp/{exp}/score/{lang}") / category
        all_file_path = cat_dir / f"{category}_all.json"
        try:
            data = _safe_read_json(all_file_path)
            all_data.append(data)
        except Exception as e:
            print(f"⚠️ 处理 {lang}/{category} 时出错: {e}")

    if not all_data:
        print(f"⚠️ 无有效数据用于平均计算: {lang}")
        return

    keys = all_data[0].keys()
    averages = {}
    for key in keys:
        values = [data.get(key, 0) for data in all_data]
        averages[key] = sum(values) / len(values)

    print(f"\n📊 {lang} 平均值:")
    for key, avg in averages.items():
        print(f"  {key}: {avg:.3f}")


# ===============================
# 主入口
# ===============================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="test_path_tts/config.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    client_cfg = cfg["client"]    
    exp = client_cfg.get("exp", {})

    langs = ["cn", "en"]
    interrupt_cata = [
        "Follow-up Questions",
        "Negation or Dissatisfaction",
        "Repetition Requests",
        "Silence or Termination",
        "Topic Switching"
    ]
    reject_rate_cata = ["Pause Handling", "Third-party Speech_before"]
    reject_resume_cata = [
        "Speech Directed at Others",
        "Third-party Speech_after",
        "User Real-time Backchannels"
    ]

    for lang in langs:
        print(f"\n处理语言目录: {lang}")
        for cat in interrupt_cata:
            try:
                build_single_category(exp, cat, lang)
            except Exception as e:
                print(f"⚠️ 处理 {lang}/{cat} 出错: {e}")

        calculate_average_of_keys(exp, interrupt_cata, lang)

        for cat in reject_rate_cata:
            try:
                reject_rate_category(exp, cat, lang)
            except Exception as e:
                print(f"⚠️ 处理 {lang}/{cat} 出错: {e}")

        for cat in reject_resume_cata:
            try:
                reject_resume_category(exp, cat, lang)
            except Exception as e:
                print(f"⚠️ 处理 {lang}/{cat} 出错: {e}")


if __name__ == "__main__":
    main()
