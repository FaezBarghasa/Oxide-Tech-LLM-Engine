import time
import sglang as sgl

def main():
    with open("bench_large_prompt.txt", "r") as f:
        base_prompt = f.read()

    prompts = [
        base_prompt + f"\n\n[USER REQUEST {i+1}]: Focus on Section {i+1} trade-offs and provide detailed microarchitectural recommendations."
        for i in range(4)
    ]

    print("Initializing SGLang Engine for Multi-User Batch Benchmark (N=4 concurrent users)...")
    llm = sgl.Engine(
        model_path="/home/jrad/models/hf_qwen2.5_0.5b_meta",
        mem_fraction_static=0.5,
        trust_remote_code=True,
        disable_cuda_graph=True
    )

    print("Engine initialized. Running warmup...")
    llm.generate(prompt="Hello", sampling_params={"temperature": 0.0, "max_new_tokens": 8})

    print(f"Running SGLang Multi-User Benchmark: 4 concurrent users x ~700 prompt tokens -> 128 completion tokens each...")
    t0 = time.perf_counter()
    outputs = llm.generate(
        prompts,
        sampling_params={
            "temperature": 0.0,
            "max_new_tokens": 128
        }
    )
    t1 = time.perf_counter()

    total_time = t1 - t0
    total_completion_tokens = sum(o.get("meta_info", {}).get("completion_tokens", 128) for o in outputs)
    total_prompt_tokens = sum(o.get("meta_info", {}).get("prompt_tokens", 700) for o in outputs)

    print("=" * 60)
    print(f"SGLang Multi-User (N=4) Results:")
    print(f"Total Wall-Clock Latency: {total_time * 1000:.2f} ms")
    print(f"Total Prompt Tokens: {total_prompt_tokens}")
    print(f"Total Completion Tokens: {total_completion_tokens}")
    print(f"Effective Aggregate Decode Speed: {total_completion_tokens / total_time:.2f} tok/s")
    print(f"Per-User Avg Latency: {total_time * 1000 / len(outputs):.2f} ms")
    for i, o in enumerate(outputs):
        print(f"User {i+1} Output (first 100 chars): {repr(o['text'][:100])}")
    print("=" * 60)
    llm.shutdown()

if __name__ == '__main__':
    main()
