import json
from pathlib import Path
from collections import Counter

def _stats_by_axis(records):
    """
    计算行为轴统计（仅用 'C' 轴），返回 ratios 里有 'C_RESPOND' 的占比。
    records: [{"key": "...", "behaviour": ["C_RESPOND"]}, ...]
    """
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
    # 保留两位小数更直观
    fmt_ratios = {ax: {k: round(v, 2) for k, v in ratios[ax].items()} for ax in ["C"]}
    avg_respond = fmt_ratios.get("C", {}).get("C_RESPOND", 0.0)
    return avg_respond

def _safe_read_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _read_behaviour_records(behaviour_path_base: Path) -> list:
    """
    兼容 {cat}_behaviour.jsonl 或 {cat}_behaviour.json
    返回 list[dict]
    """
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
        # 可能是列表或字典；尽量兜底为列表
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            return [data]
    raise FileNotFoundError(f"行为评估文件不存在: {jsonl} 或 {js}")

def run_build_category_all(exp: str, categories: list, cat2group: dict):
    """
    读取每个 category 的三个结果文件，整合成:
    exp/{exp}/score/category_all.json

    参数:
    - exp: 例如 "exp1"
    - categories: 如 ["Follow-up Questions", "Reject Case A", ...]
    - cat2group: 手动指定每个 category 属于 "interrupt" 或 "reject"
        形如: {"Follow-up Questions": "interrupt", "Reject Something": "reject"}

    读取的文件（每个 category 目录里）:
      1) {category}_latency_results.json  -> average_latency.{avg_latency_stop, avg_latency_resp}
      2) {category}_ftd.json              -> avg_first_time_delay
      3) {category}_behaviour.jsonl/json  -> 统计 average_RESPOND_score
    """
    base_score_dir = Path(f"exp/{exp}/score")
    result = {"interrupt": {}, "reject": {}}

    for category in categories:
        group = cat2group.get(category)
        if group not in ("interrupt", "reject"):
            raise ValueError(f"cat2group 未正确指定 {category} 的分组，应为 'interrupt' 或 'reject'。")

        cat_dir = base_score_dir / category

        # 1) latency
        latency_path = cat_dir / f"{category}_latency_results.json"
        latency_data = _safe_read_json(latency_path)
        avg_latency_stop = latency_data.get("average_latency", {}).get("avg_latency_stop", None)
        avg_latency_resp = latency_data.get("average_latency", {}).get("avg_latency_resp", None)

        # 2) ftd
        ftd_path = cat_dir / f"{category}_ftd.json"
        ftd_data = _safe_read_json(ftd_path)
        avg_first_time_delay = ftd_data.get("avg_first_time_delay", None)

        # 3) behaviour -> average_RESPOND_score
        behaviour_base = cat_dir / f"{category}_content_tags.json"
        records = _read_behaviour_records(behaviour_base)
        average_RESPOND_score = _stats_by_axis(records)

        # 写入对应第一层键下的第二层 category
        result[group][category] = {
            "avg_latency_stop": avg_latency_stop,
            "avg_latency_resp": avg_latency_resp,
            "avg_first_time_delay": avg_first_time_delay,
            "average_RESPOND_score": average_RESPOND_score
        }

    out_path = base_score_dir / "category_all.json"
    base_score_dir.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"category_all.json 已生成: {out_path}")


def main():
        # 你的 categories
    exp = "exp1"
    categories = ["Follow-up Questions"]

    # 你手动决定每个 category 属于 interrupt 还是 reject
    cat2group = {
        "Follow-up Questions": "interrupt",
        # 例如将来还有别的：
        # "Refusal Cases": "reject",
    }

    run_build_category_all(exp, categories, cat2group)
if __name__ == "__main__":
    main()