# 评分流程说明文档
Full Duplex Conversation Evaluation – Scoring Specification

---

# 1. 总览

评分体系包含三个核心模块：

- Interruption（打断类）
- Rejection（拒识类）
- Latency（延迟类）

最终输出字段如下：

| 字段名 | 含义 |
|--------|------|
| Interruption Total Score | 打断响应质量（百分制） |
| Rejection Total Score | 拒识行为质量（百分制） |
| First Response Delay | 首次响应延迟（秒） |
| avg_latency_stop | 停止延迟（秒） |
| avg_latency_resp | 回应延迟（秒） |
| Total Delay | 综合延迟（秒） |

输入数据文件格式位置：  
exp/{exp_id}/score/{lang}/{category}/{category}_all.json  
其中 lang ∈ {cn, en}

---

# 2. JSON 字段说明

不同文件可能包含如下字段：

| 字段名 | 描述 |
|--------|------|
| average_RESPOND_score | 打断任务 RESPOND 得分 |
| average_RESUME_score | 拒识任务 RESUME 得分 |
| reject_rate | 拒识率 |
| avg_latency_stop | 停止延迟 |
| avg_latency_resp | 回应延迟 |
| avg_first_time_delay | 首次响应延迟 |

---

# 3. Interruption（打断类）评分规则

## 3.1 参与类别（共 10 文件）

包括以下 5 类，每类有 cn 和 en 两个文件：

- Follow-up Questions
- Negation or Dissatisfaction
- Repetition Requests
- Silence or Termination
- Topic Switching

共 10 个文件。

## 3.2 指标计算方式

    Interruption Total Score  
    = 所有 average_RESPOND_score 的简单平均 × 100  
    （共 10 个值）

    avg_latency_stop  
    = 所有 avg_latency_stop 数值的简单平均

    avg_latency_resp  
    = 所有 avg_latency_resp 数值的简单平均

    avg_first_time_delay（interruption 部分）  
    = 所有 avg_first_time_delay 数值的简单平均

---

# 4. Rejection（拒识类）评分（自定义逻辑）

参与的五类 × 中英文 = 10 文件：

| 类别 | 指标 |
|------|------|
| Speech Directed at Others | average_RESUME_score |
| Third-party Speech_after | average_RESUME_score |
| User Real-time Backchannels | average_RESUME_score |
| Pause Handling | reject_rate |
| Third-party Speech_before | reject_rate |

你的自定义评分步骤如下：

## 步骤 1：对每类进行中英文平均

SDA = mean(SDA_cn, SDA_en)  
URBC = mean(URBC_cn, URBC_en)  
PH = mean(PH_cn, PH_en)  
before = mean(before_cn, before_en)  
after = mean(after_cn, after_en)

## 步骤 2：合并第三方 before/after

third_party = (before + after) / 2

## 步骤 3：四项一起做均值 → 拒识总分

Rejection Total Score = mean([SDA, URBC, PH, third_party]) × 100

---

# 5. 延迟类（Latency）评分规则

延迟包含三个部分：

- First Response Delay
- avg_latency_stop（来自 Interruption）
- avg_latency_resp（来自 Interruption）

---

# 5.1 First Response Delay

严格遵循你的要求：

First Response Delay  
= 所有类别（10 类 × 中英 = 20 文件）的 avg_first_time_delay 的简单平均

不做加权。

示例值：1.528

---

# 5.2 Total Delay

    Total Delay  
    = mean([avg_latency_stop, avg_latency_resp, First Response Delay])

    示例：  
    (1.106 + 2.461 + 1.528) / 3 = 1.698

---

# 6. 最终输出格式（示意）

以下为最终 all.json 的结构示例：

    {
    "interrupt": {
        "Interruption Total Score": 89.7,
        "avg_latency_stop": 1.106,
        "avg_latency_resp": 2.461,
        "avg_first_time_delay": 1.679
    },
    "reject": {
        "Speech Directed at Others": {
        "average_RESUME_score": 0.235,
        "avg_first_time_delay": 1.8
        },
        "Third-party Speech_after": {
        "average_RESUME_score": 0.34,
        "avg_first_time_delay": 1.72
        },
        "User Real-time Backchannels": {
        "average_RESUME_score": 0.765,
        "avg_first_time_delay": 1.536
        },
        "Pause Handling": {
        "reject_rate": 0.83,
        "avg_first_time_delay": 1.826
        },
        "Third-party Speech_before": {
        "reject_rate": 0.0,
        "avg_first_time_delay": 0
        }
    },
    "First Response Delay": 1.528,
    "Interruption Total Score": 89.7,
    "Rejection Total Score": 50.0,
    "Total Delay": 1.698
    }

---

# 7. 验证清单

验证评分正确性时需检查：

1. 是否读取了全部 20 个 _all.json 文件  
2. Interruption 部分：10 个 RESPOND score 是否全部参与平均  
3. Rejection 部分：是否按你的规则进行  
   - 每类先中英文平均  
   - 合并 before/after  
   - 四类均值 ×100  
4. First Response Delay 是否包含全部 20 个 ftd 值  
5. Total Delay 是否为 stop、resp、FTD 的均值  
6. 各字段均无加权处理  

---

# 8. 使用说明

此文档用于：

- 评分脚本（inter_score.py / ave.py）  
- 评分体系定义  
- README 与项目文档  
- 论文附录  
- 内部评测流程说明

如需英文版、流程图（mermaid/draw.io）、或自动验证脚本，请告知。
