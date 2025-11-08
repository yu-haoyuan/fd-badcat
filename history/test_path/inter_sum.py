import os
import json
from pathlib import Path

def merge_scores(json_paths, category, save_path):
    """
    读取多个 JSON 文件，合并为一个指定格式的输出 JSON。
    参数：
      json_paths: list[str] 或 list[Path]，输入的 json 文件路径
      category: str，类别名（如 follow_up）
      save_path: 输出文件路径
    """
    data = {}

    # 遍历每个 JSON 文件
    for path in json_paths:
        with open(path, "r", encoding="utf-8") as f:
            content = json.load(f)

        # 示例：假设每个文件包含一组分数字段
        # 你可以改成自己 JSON 文件里的实际键名
        data.update(content)

    # 构造输出结构
    result = {
        category: {
            "average_RESPOND_score": data.get("average_RESPOND_score", None),
            "average_stop_latency": data.get("average_stop_latency", None),
            "average_response_latency": data.get("average_response_latency", None),
            "First_Response_Delay": data.get("First_Response_Delay", None),
            "Interruption_score": data.get("Interruption_score", None)
        }
    }

    # 保存输出
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"✅ 已保存到: {save_path}")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    # ===== 示例用法 =====
    # 3 个输入文件
    json_files = [
        "score/interruption.json",
        "score/response.json",
        "score/latency.json"
    ]
    # 输出类别
    category = "follow_up"
    # 输出路径
    save_path = "score/merged/follow_up.json"

    merge_scores(json_files, category, save_path)
