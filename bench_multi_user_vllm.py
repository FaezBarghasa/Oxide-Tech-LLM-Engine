import time
import asyncio
from vllm import LLM, SamplingParams

def main():
    with open("bench_large_prompt.txt", "r") as f:
        base_prompt = f.read()

    # Create 4 concurrent user requests with large prompts
    # Slight variations to simulate authentic concurrent users
    prompts = [
        base_prompt + f"\n\n[USER REQUEST {i+1}]: Focus on Section {i+1} trade-offs and provide detailed microarchitectural recommendations."
        for i in range(4)
    ]

    print("Initializing vLLM Engine for Multi-User Batch Benchmark (N=4 concurrent users)...")
    llm = LLM(
        model="/home/jrad/models/hf_qwen2.5_0.5b_meta",
        gpu_memory_utilization=0.6,
        trust_remote_code=True,
        enforce_eager=True,
        max_model_len=2048,
    )

    print("Engine initialized. Running warmup...")
    llm.generate(["Hello"], SamplingParams(temperature=0.0, max_tokens=8))

    print(f"Running Multi-User Benchmark: 4 concurrent users x ~700 prompt tokens -> 128 completion tokens each...")
    t0 = time.perf_counter()
    outputs = llm.generate(
        prompts,
        SamplingParams(temperature=0.0, max_tokens=128)
    )
    t1 = time.perf_counter()

    total_time = t1 - t0
    total_prompt_tokens = sum(len(o.prompt_token_ids) for o in outputs)
    total_completion_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)

    print("=" * 60)
    print(f"vLLM Multi-User (N=4) Results:")
    print(f"Total Wall-Clock Latency: {total_time * 1000:.2f} ms")
    print(f"Total Prompt Tokens: {total_prompt_tokens}")
    print(f"Total Completion Tokens: {total_completion_tokens}")
    print(f"Effective Aggregate Decode Speed: {total_completion_tokens / total_time:.2f} tok/s")
    print(f"Per-User Avg Latency: {total_time * 1000 / len(outputs):.2f} ms")
    for i, o in enumerate(outputs):
        print(f"User {i+1} Output (first 100 chars): {repr(o.outputs[0].text[:100])}")
    print("=" * 60)

if __name__ == "__main__":
    main()
