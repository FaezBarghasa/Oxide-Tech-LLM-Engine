import time
import sglang as sgl

def main():
    with open("bench_large_prompt.txt", "r") as f:
        prompt = f.read()

    print("Initializing SGLang Engine...")
    llm = sgl.Engine(
        model_path="/home/jrad/models/hf_qwen2.5_0.5b_meta",
        mem_fraction_static=0.5,
        trust_remote_code=True,
        disable_cuda_graph=True
    )

    print("Engine initialized. Running warmup...")
    res = llm.generate(prompt="Hello", sampling_params={"temperature": 0.0, "max_new_tokens": 8})
    print("Warmup finished. Generated:", res["text"])

    print("Running Large Prompt Benchmark (689 tokens -> 128 tokens)...")
    t0 = time.perf_counter()
    output = llm.generate(
        prompt=prompt,
        sampling_params={
            "temperature": 0.0,
            "max_new_tokens": 128
        }
    )
    t1 = time.perf_counter()

    total_time = t1 - t0
    out_text = output["text"]
    meta = output.get("meta_info", {})
    print("=" * 60)
    print(f"Generated text: {repr(out_text[:200])}...")
    print(f"Total Latency: {total_time * 1000:.2f} ms")
    print(f"Meta Info: {meta}")
    num_tokens = meta.get("completion_tokens", 128)
    print(f"Effective E2E Token Generation Speed: {num_tokens / total_time:.2f} tok/s")
    print("=" * 60)
    llm.shutdown()

if __name__ == '__main__':
    main()
