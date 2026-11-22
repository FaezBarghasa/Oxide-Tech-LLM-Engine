import time
from vllm import LLM, SamplingParams

def main():
    with open("bench_large_prompt.txt", "r") as f:
        prompt = f.read()

    print("Initializing vLLM Engine...")
    llm = LLM(
        model="/home/jrad/models/hf_qwen2.5_0.5b_meta",
        gpu_memory_utilization=0.6,
        trust_remote_code=True,
        enforce_eager=True,
        max_model_len=2048,
    )

    print("Engine initialized. Running warmup...")
    llm.generate(["Hello"], SamplingParams(temperature=0.0, max_tokens=8))

    print("Running Large Prompt Benchmark (689 tokens -> 128 tokens)...")
    t0 = time.perf_counter()
    outputs = llm.generate(
        [prompt],
        SamplingParams(temperature=0.0, max_tokens=128)
    )
    t1 = time.perf_counter()

    total_time = t1 - t0
    out = outputs[0]
    prompt_len = len(out.prompt_token_ids)
    output_len = len(out.outputs[0].token_ids)
    out_text = out.outputs[0].text

    print("=" * 60)
    print(f"Generated text: {repr(out_text[:200])}...")
    print(f"Prompt Tokens: {prompt_len}")
    print(f"Completion Tokens: {output_len}")
    print(f"Total Latency: {total_time * 1000:.2f} ms")
    print(f"Effective E2E Token Generation Speed: {output_len / total_time:.2f} tok/s")
    print("=" * 60)

if __name__ == "__main__":
    main()
