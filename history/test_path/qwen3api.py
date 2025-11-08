import os
from openai import OpenAI
"sk-553f353ca3d1436b9ec9a9c728e30958"
client = OpenAI(
    # 新加坡和北京地域的API Key不同。获取API Key：https://www.alibabacloud.com/help/zh/model-studio/get-api-key
    # api_key="sk-553f353ca3d1436b9ec9a9c728e30958",
    # 以下为新加坡地域url，若使用北京地域的模型，需将url替换为：https://dashscope.aliyuncs.com/compatible-mode/v1
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

completion = client.chat.completions.create(
    model="qwen3-omni-flash",
    messages=[{"role": "user", "content": "你是谁"}],
    modalities=["text"],
    audio={"voice": "Cherry", "format": "wav"},
    stream=True,
    stream_options={"include_usage": True},
)

text_output = ""
for chunk in completion:
    if chunk.choices:
        delta = chunk.choices[0].delta
        if hasattr(delta, "content") and delta.content:
            text_output += delta.content
    # elif hasattr(chunk, "usage"):
    #     print(chunk.usage)  # 如果需要可以保留 usage 信息

print(text_output)