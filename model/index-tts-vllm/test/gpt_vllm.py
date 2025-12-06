import asyncio
import random
import time
import uuid
import torch
import os
from typing import AsyncGenerator, List, Dict, Any
import numpy as np

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # # 以下设置可能会轻微影响性能，但确保完全可复现
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

set_seed(42)

import sys
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

import patch_vllm  # ⚠️ Monkey Patch, do not delete this line

from indextts.gpt.index_tts_gpt2_vllm_v1 import PLACEHOLDER_TOKEN, PLACEHOLDER_TOKEN_ID

from vllm import SamplingParams, TokensPrompt
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.v1.engine.async_llm import AsyncLLM

model_dir = os.path.join(root_dir, "checkpoints/IndexTTS-2-vLLM")

vllm_dir = os.path.join(model_dir, "gpt")

async def run_single_inference(llm: AsyncLLM, inputs_embeds: torch.Tensor):
    request_id = uuid.uuid4().hex

    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=0.8,
        top_k=30,
        repetition_penalty=10.0,
        max_tokens=random.randint(400, 1200),  # 8s - 24s # 1818
        ignore_eos=True,
        stop_token_ids=[],  # 8193
    )
    
    multi_modal_data = {"audio": {"audio_embeds": [inputs_embeds.squeeze(0).cpu()]}}
    fake_inputs = PLACEHOLDER_TOKEN * 1
    tokens_prompt = TokensPrompt(prompt=fake_inputs, multi_modal_data=multi_modal_data)

    start_time = time.time()
    output_generator = llm.generate(tokens_prompt, sampling_params, request_id=request_id)
    
    prefill_flag = True
    first_token_time = None
    ttft = 0.0
    output_tokens = []

    async for output in output_generator:
        if prefill_flag:
            first_token_time = time.time()
            ttft = first_token_time - start_time
            prefill_flag = False

        final_output = output

    end_time = time.time()
    
    latency = end_time - start_time
    decode_time = end_time - first_token_time
    
    generated_tokens = final_output.outputs[0].token_ids[:-2]
    num_generated_tokens = len(generated_tokens)
    
    decode_speed = 0
    if decode_time > 0:
        decode_speed = num_generated_tokens / decode_time

    return {
        "prefill_time": ttft,
        "decode_time": decode_time,
        "latency": latency,
        "num_generated_tokens": num_generated_tokens,
        "decode_speed_tokens_per_sec": decode_speed,
    }

async def run_user_simulation(llm: AsyncLLM, inputs_embeds: torch.Tensor, num_runs: int = 10) -> List[Dict[str, Any]]:
    user_results = []
    for _ in range(num_runs):
        result = await run_single_inference(llm, inputs_embeds)
        user_results.append(result)
    return user_results

async def benchmark(llm_engine, concurrency_levels, runs_per_user):
    hidden_dim = 1280
    conds_len = 34
    max_text_tokens = 200

    # warm up
    fake_inputs_embeds = torch.randn(
        1, 
        conds_len + max_text_tokens,
        hidden_dim, 
        dtype=torch.float16,
        device="cpu"
    )
    await run_single_inference(llm_engine, fake_inputs_embeds) 

    for concurrency in concurrency_levels:
        # 为每个并发用户准备独立的输入数据
        user_inputs = [torch.randn(
            1, 
            conds_len + max_text_tokens,
            hidden_dim, 
            dtype=torch.float16, 
            device="cpu"
        ) for _ in range(concurrency)]
        
        # 创建并发任务，每个任务代表一个连续请求10次的用户
        tasks = [
            run_user_simulation(llm_engine, user_inputs[c_idx], runs_per_user) 
            for c_idx in range(concurrency)
        ]
        
        benchmark_start_time = time.time()
        results = await asyncio.gather(*tasks)
        benchmark_total_time = time.time() - benchmark_start_time

        # --- 统计和计算结果 ---
        all_ttfts_ms = []
        all_latencies_ms = []
        all_num_generated_tokens = []
        total_requests = 0
        total_generated_tokens = 0

        for user_results in results:
            for res in user_results:
                total_requests += 1
                total_generated_tokens += res["num_generated_tokens"]
                all_ttfts_ms.append(res["prefill_time"] * 1000)
                all_latencies_ms.append(res["latency"] * 1000)
                all_num_generated_tokens.append(res["num_generated_tokens"])

        # Convert to numpy arrays for statistics
        ttfts_np = np.array(all_ttfts_ms)
        latencies_np = np.array(all_latencies_ms)
        tokens_np = np.array(all_num_generated_tokens)
        
        # 总吞吐量 = 在总测试时间内生成的总token数
        total_throughput = total_generated_tokens / benchmark_total_time

        print(f"## Concurrency Level: {concurrency}")
        print(f"\n*   **Total Requests:** {concurrency * runs_per_user}")
        print(f"*   **Total Time:** {benchmark_total_time:.2f} s")
        print(f"*   **Total Throughput:** {total_throughput:.2f} tokens/s\n")

        print("| Metric                 | Min      | Max       | Mean      | P50       | P95       | P99       |")
        print("|------------------------|----------|-----------|-----------|-----------|-----------|-----------|")
        
        if total_requests > 0:
            # TTFT
            print(f"| TTFT (ms)              | {np.min(ttfts_np):<8.2f} | {np.max(ttfts_np):<9.2f} | {np.mean(ttfts_np):<9.2f} | {np.percentile(ttfts_np, 50):<9.2f} | {np.percentile(ttfts_np, 95):<9.2f} | {np.percentile(ttfts_np, 99):<9.2f} |")
    
            # Latency
            print(f"| Latency (ms)           | {np.min(latencies_np):<8.2f} | {np.max(latencies_np):<9.2f} | {np.mean(latencies_np):<9.2f} | {np.percentile(latencies_np, 50):<9.2f} | {np.percentile(latencies_np, 95):<9.2f} | {np.percentile(latencies_np, 99):<9.2f} |")
    
            # Num Generated Tokens
            print(f"| Num Generated Tokens   | {np.min(tokens_np):<8.0f} | {np.max(tokens_np):<9.0f} | {np.mean(tokens_np):<9.2f} | {np.percentile(tokens_np, 50):<9.0f} | {np.percentile(tokens_np, 95):<9.0f} | {np.percentile(tokens_np, 99):<9.0f} |")
        print("\n" + "-"*40 + "\n")


if __name__ == "__main__":
    gpu_memory_utilization = 0.5  # 0.25
    concurrency_levels = [1, 4, 8, 16, 32, 64]  # 并发数
    runs_per_user = 10  # 每个并发的请求数

    engine_args = AsyncEngineArgs(
        model=vllm_dir,
        tensor_parallel_size=1,
        dtype="auto",
        gpu_memory_utilization=gpu_memory_utilization,
        max_num_seqs=50
    )
    llm_engine: AsyncLLM = AsyncLLM.from_engine_args(engine_args)

    asyncio.run(benchmark(llm_engine, concurrency_levels, runs_per_user))