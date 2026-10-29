use clap::Parser;
use memmap2::Mmap;
use oxide_models::{GgufFile, GgufTokenizer};
use std::collections::HashMap;
use std::fs::File;
use std::path::PathBuf;
use std::process::Command;

#[derive(Parser, Debug)]
#[command(
    name = "bench-matrix",
    about = "Empirical Benchmark Matrix & Outside Judge: Oxide vs llama.cpp"
)]
struct Args {
    #[arg(
        short,
        long,
        default_value = "/home/jrad/models/Huihui-Qwen3-8B-abliterated-v2.Q4_K_M.gguf"
    )]
    model: PathBuf,

    #[arg(long, default_value = "16,64,512")]
    prompts: String,

    #[arg(long, default_value_t = 128)]
    gen_tokens: usize,

    #[arg(long, default_value_t = 5)]
    iterations: usize,

    #[arg(long, default_value = "target/bench_matrix.json")]
    report: PathBuf,
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Default, Clone)]
struct Telemetry {
    temp_c: f64,
    sm_clock_mhz: f64,
    power_w: f64,
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Default, Clone)]
struct BenchCell {
    test: String,
    llamacpp_tps: f64,
    oxide_median_tps: f64,
    oxide_p95_tps: f64,
    speedup_ratio: f64,
    samples: Vec<f64>,
    telemetry_before: Option<Telemetry>,
    telemetry_after: Option<Telemetry>,
    passed: bool,
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Default)]
struct CorrectnessJudge {
    prompt: String,
    llamacpp_output: String,
    oxide_output: String,
    agreement_percent: f64,
    passed: bool,
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Default)]
struct BenchReport {
    model: String,
    device: String,
    judge: CorrectnessJudge,
    cells: Vec<BenchCell>,
}

const CORPUS: &str = "\
The history of computing hardware covers the developments from early simple devices to aid calculation \
to modern day supercomputers and neural processing units. Before the 20th century, most calculations were done \
by human computers. The first aids to computation were purely mechanical devices which required the operator \
to set the initial values of an elementary arithmetic operation, then manipulate the device to obtain the result. \
Later, computers represented numbers in a continuous form, for instance distance along a scale, rotation of a shaft, \
or a size of an electrical voltage. Numbers could also be represented in the form of digits, automatically manipulated \
by a mechanical mechanism. Although this approach required more complex mechanisms, it greatly increased the precision of results. \
The invention of the transistor and the integrated circuit led to remarkable leaps in computational capacity and efficiency, \
ultimately enabling modern artificial intelligence accelerators with specialized tensor processing units capable of \
executing trillions of operations per second. Modern machine learning architectures, specifically transformer-based large \
language models, require specialized low-precision matrix arithmetic including 4-bit and 8-bit quantized representations \
to maximize memory bandwidth efficiency and operational throughput during high-concurrency inference workloads across \
physical semiconductor platforms including NVIDIA Ada Lovelace GPUs. Deep learning workloads demand highly optimized \
kernels running on unified memory architectures with tight integration between execution pipelines and matrix multiply-accumulate \
hardware instructions to achieve maximum sustained arithmetic intensity.";

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let args = Args::parse();
    println!("==================================================");
    println!("  OXIDE vs LLAMA.CPP EMPIRICAL BENCHMARK MATRIX   ");
    println!("==================================================");
    println!("Model       : {}", args.model.display());
    println!("Prompt Lens : {}", args.prompts);
    println!("Gen Tokens  : {}", args.gen_tokens);
    println!("Iterations  : {} (warmup + 5 runs)", args.iterations);
    println!("Report Path : {}", args.report.display());
    println!("==================================================");

    // Initialize Tokenizer from GGUF
    let tokenizer = {
        let file = File::open(&args.model)?;
        // SAFETY: The model file is mapped read-only for metadata extraction.
        let mmap = unsafe { Mmap::map(&file)? };
        let gguf = GgufFile::parse(&mmap)?;
        GgufTokenizer::from_gguf(&gguf)
    };

    let mut report = BenchReport {
        model: args.model.to_string_lossy().to_string(),
        device: "NVIDIA GeForce RTX 4060 Laptop dGPU".to_string(),
        judge: CorrectnessJudge::default(),
        cells: Vec::new(),
    };

    // 0. Correctness Judge: Verify greedy output matches llama.cpp on a standard reasoning prompt
    println!("\n[1/3] Running Outside Correctness Judge...");
    let judge_prompt = "Count from 1 to 10:";
    report.judge.prompt = judge_prompt.to_string();

    let llama_judge_out = Command::new("llama-cli")
        .args([
            "-st",
            "--simple-io",
            "--no-display-prompt",
            "-m",
            &args.model.to_string_lossy(),
            "-p",
            judge_prompt,
            "-n",
            "16",
            "-ngl",
            "99",
            "--temp",
            "0",
        ])
        .output();

    if let Ok(out) = llama_judge_out {
        let text = String::from_utf8_lossy(&out.stdout);
        report.judge.llamacpp_output = parse_llama_generated_text(&text);
    }

    let oxide_judge_out = Command::new("target/release/oxide-engine")
        .args([
            "-m",
            &args.model.to_string_lossy(),
            "--n-gpu-layers",
            "99",
            "-b",
            "cuda",
            "-p",
            judge_prompt,
            "-n",
            "16",
            "--greedy",
            "--chat-format",
            "raw",
            "--speculative",
            "none",
        ])
        .output();

    if let Ok(out) = oxide_judge_out {
        let text = String::from_utf8_lossy(&out.stdout);
        report.judge.oxide_output = parse_oxide_generated_text(&text);
    }

    report.judge.agreement_percent =
        compute_text_similarity(&report.judge.llamacpp_output, &report.judge.oxide_output);
    report.judge.passed = report.judge.agreement_percent >= 70.0;

    println!("  Llama.cpp Output : {:?}", report.judge.llamacpp_output);
    println!("  Oxide Output     : {:?}", report.judge.oxide_output);
    println!(
        "  Token Agreement  : {:.2}% ({})",
        report.judge.agreement_percent,
        if report.judge.passed { "PASS" } else { "FAIL" }
    );

    // 1. Run llama-bench outside judge
    println!("\n[2/3] Invoking llama-bench judge (3 runs)...");
    let llama_output = Command::new("llama-bench")
        .args([
            "-m",
            &args.model.to_string_lossy(),
            "-p",
            &args.prompts,
            "-n",
            &args.gen_tokens.to_string(),
            "-ngl",
            "99",
            "-r",
            "3",
        ])
        .output()?;

    let llama_stdout = String::from_utf8_lossy(&llama_output.stdout);
    println!("{llama_stdout}");

    let mut llama_scores = HashMap::new();
    for line in llama_stdout.lines() {
        if line.contains("pp") || line.contains("tg") {
            let parts: Vec<&str> = line.split('|').map(|s| s.trim()).collect();
            if parts.len() >= 8 {
                let test_name = parts[6];
                let tps_str = parts[7].split('±').next().unwrap_or("0").trim();
                if let Ok(v) = tps_str.parse::<f64>() {
                    llama_scores.insert(test_name.to_string(), v);
                }
            }
        }
    }

    // 2. Run Oxide on each prefill cell with exact real text prompts
    println!("\n[3/3] Invoking Oxide-Engine benchmark runs with GPU telemetry...");
    let prompt_lens: Vec<usize> = args
        .prompts
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();

    // Prepare large corpus tokens
    let mut corpus_tokens = tokenizer.encode(CORPUS);
    while corpus_tokens.len() < 1024 {
        corpus_tokens.extend_from_slice(&tokenizer.encode(CORPUS));
    }

    for p in &prompt_lens {
        let test_name = format!("pp{p}");
        let llama_tps = *llama_scores.get(&test_name).unwrap_or(&0.0);

        // Slice exact tokens from real text corpus
        let target_len = *p;
        let token_slice = &corpus_tokens[..target_len.min(corpus_tokens.len())];
        let p_str = tokenizer.decode(token_slice);

        let t_before = get_gpu_telemetry();

        // 1 warmup run
        let _ = run_oxide_prefill(&args.model, &p_str);

        // N benchmark runs
        let mut samples = Vec::new();
        for _ in 0..args.iterations {
            let tps = run_oxide_prefill(&args.model, &p_str);
            if tps > 0.0 {
                samples.push(tps);
            }
        }

        let t_after = get_gpu_telemetry();
        let (median_tps, p95_tps) = compute_stats(samples.clone());
        let ratio = if llama_tps > 0.0 {
            median_tps / llama_tps
        } else {
            0.0
        };

        report.cells.push(BenchCell {
            test: test_name,
            llamacpp_tps: llama_tps,
            oxide_median_tps: median_tps,
            oxide_p95_tps: p95_tps,
            speedup_ratio: ratio,
            samples,
            telemetry_before: t_before,
            telemetry_after: t_after,
            passed: ratio >= 1.0,
        });
    }

    // AR Decode test tg
    let test_name = format!("tg{}", args.gen_tokens);
    let llama_tps = *llama_scores.get(&test_name).unwrap_or(&0.0);
    let decode_prompt = "Count to 100:";

    let t_before = get_gpu_telemetry();
    let _ = run_oxide_decode(&args.model, decode_prompt, args.gen_tokens, "none");
    let mut samples = Vec::new();
    for _ in 0..args.iterations {
        let tps = run_oxide_decode(&args.model, decode_prompt, args.gen_tokens, "none");
        if tps > 0.0 {
            samples.push(tps);
        }
    }
    let t_after = get_gpu_telemetry();
    let (median_tps, p95_tps) = compute_stats(samples.clone());
    let ratio = if llama_tps > 0.0 {
        median_tps / llama_tps
    } else {
        0.0
    };

    report.cells.push(BenchCell {
        test: test_name,
        llamacpp_tps: llama_tps,
        oxide_median_tps: median_tps,
        oxide_p95_tps: p95_tps,
        speedup_ratio: ratio,
        samples,
        telemetry_before: t_before,
        telemetry_after: t_after,
        passed: ratio >= 0.95,
    });

    // Speculative Decode test tg-spec
    let spec_test_name = format!("tg{}-spec", args.gen_tokens);
    let spec_prompt = "Write a clear, concise summary of quantum mechanics principles:";

    let t_before = get_gpu_telemetry();
    let _ = run_oxide_decode(&args.model, spec_prompt, args.gen_tokens, "auto");
    let mut samples_spec = Vec::new();
    for _ in 0..args.iterations {
        let tps = run_oxide_decode(&args.model, spec_prompt, args.gen_tokens, "auto");
        if tps > 0.0 {
            samples_spec.push(tps);
        }
    }
    let t_after = get_gpu_telemetry();
    let (median_spec_tps, p95_spec_tps) = compute_stats(samples_spec.clone());
    let spec_ratio = if llama_tps > 0.0 {
        median_spec_tps / llama_tps
    } else {
        0.0
    };

    report.cells.push(BenchCell {
        test: spec_test_name,
        llamacpp_tps: llama_tps,
        oxide_median_tps: median_spec_tps,
        oxide_p95_tps: p95_spec_tps,
        speedup_ratio: spec_ratio,
        samples: samples_spec,
        telemetry_before: t_before,
        telemetry_after: t_after,
        passed: spec_ratio >= 1.0,
    });

    // Save JSON
    if let Some(parent) = args.report.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let json_str = serde_json::to_string_pretty(&report)?;
    std::fs::write(&args.report, json_str)?;

    println!(
        "\n=================================================================================="
    );
    println!("                           EMPIRICAL COMPARISON SCOREBOARD                        ");
    println!("==================================================================================");
    println!(
        "| Test | llama.cpp (t/s) | Oxide Med (t/s) | Oxide p95 (t/s) | Speedup Ratio | Status |"
    );
    println!("| :--- | :---: | :---: | :---: | :---: | :---: |");
    for cell in &report.cells {
        let status = if cell.passed { "PASS" } else { "FAIL" };
        println!(
            "| {:<8} | {:>15.2} | {:>15.2} | {:>15.2} | {:>12.2}x | {:^6} |",
            cell.test,
            cell.llamacpp_tps,
            cell.oxide_median_tps,
            cell.oxide_p95_tps,
            cell.speedup_ratio,
            status
        );
    }
    println!("==================================================================================");

    Ok(())
}

fn run_oxide_prefill(model: &std::path::Path, prompt: &str) -> f64 {
    let out = Command::new("target/release/oxide-engine")
        .args([
            "-m",
            &model.to_string_lossy(),
            "--n-gpu-layers",
            "99",
            "-b",
            "cuda",
            "-p",
            prompt,
            "-n",
            "1",
            "--greedy",
            "--chat-format",
            "raw",
            "--speculative",
            "none",
        ])
        .output();

    if let Ok(o) = out {
        let stdout = String::from_utf8_lossy(&o.stdout);
        parse_prefill_tps(&stdout)
    } else {
        0.0
    }
}

fn run_oxide_decode(
    model: &std::path::Path,
    prompt: &str,
    gen_tokens: usize,
    speculative: &str,
) -> f64 {
    let out = Command::new("target/release/oxide-engine")
        .args([
            "-m",
            &model.to_string_lossy(),
            "--n-gpu-layers",
            "99",
            "-b",
            "cuda",
            "-p",
            prompt,
            "-n",
            &gen_tokens.to_string(),
            "--greedy",
            "--chat-format",
            "raw",
            "--speculative",
            speculative,
        ])
        .output();

    if let Ok(o) = out {
        let stdout = String::from_utf8_lossy(&o.stdout);
        parse_decode_tps(&stdout)
    } else {
        0.0
    }
}

fn get_gpu_telemetry() -> Option<Telemetry> {
    let out = Command::new("nvidia-smi")
        .args([
            "--query-gpu=temperature.gpu,clocks.current.sm,power.draw",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;
    let stdout = String::from_utf8_lossy(&out.stdout);
    let parts: Vec<&str> = stdout.trim().split(',').map(|s| s.trim()).collect();
    if parts.len() >= 3 {
        let temp_c = parts[0].parse().unwrap_or(0.0);
        let sm_clock_mhz = parts[1].parse().unwrap_or(0.0);
        let power_w = parts[2].parse().unwrap_or(0.0);
        Some(Telemetry {
            temp_c,
            sm_clock_mhz,
            power_w,
        })
    } else {
        None
    }
}

fn compute_stats(mut samples: Vec<f64>) -> (f64, f64) {
    if samples.is_empty() {
        return (0.0, 0.0);
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = samples[samples.len() / 2];
    let p95_idx = ((samples.len() as f64) * 0.95).ceil() as usize;
    let p95 = samples[(p95_idx.saturating_sub(1)).min(samples.len() - 1)];
    (median, p95)
}

fn compute_text_similarity(s1: &str, s2: &str) -> f64 {
    let w1: Vec<&str> = s1.split_whitespace().collect();
    let w2: Vec<&str> = s2.split_whitespace().collect();
    if w1.is_empty() || w2.is_empty() {
        return 0.0;
    }
    let mut matches = 0;
    let min_len = w1.len().min(w2.len());
    for i in 0..min_len {
        if w1[i] == w2[i] {
            matches += 1;
        }
    }
    (matches as f64) / (w1.len().max(w2.len()) as f64) * 100.0
}

fn parse_llama_generated_text(stdout: &str) -> String {
    let mut capturing = false;
    let mut lines = Vec::new();
    for line in stdout.lines() {
        if line.starts_with("> ") {
            capturing = true;
            continue;
        }
        if capturing {
            if line.contains("[ Prompt:") || line.starts_with("Exiting...") {
                break;
            }
            lines.push(line.trim());
        }
    }
    lines.join(" ").trim().to_string()
}

fn parse_oxide_generated_text(stdout: &str) -> String {
    let mut capturing = false;
    let mut lines: Vec<String> = Vec::new();
    for raw_line in stdout.lines() {
        let line = strip_ansi_escapes(raw_line);
        if line.starts_with("Assistant:") {
            capturing = true;
            let rem = line.trim_start_matches("Assistant:").trim();
            if !rem.is_empty() && !rem.contains("INFO") {
                lines.push(rem.to_string());
            }
            continue;
        }
        if capturing {
            if line.starts_with("--------------------------------------------------") {
                break;
            }
            // Skip tracing logs
            if line.contains(" INFO ")
                || line.contains(" WARN ")
                || line.contains(" ERROR ")
                || (line.len() > 20
                    && line.chars().nth(4) == Some('-')
                    && line.chars().nth(7) == Some('-'))
            {
                continue;
            }
            let trimmed = line.trim();
            if !trimmed.is_empty() {
                lines.push(trimmed.to_string());
            }
        }
    }
    lines.join(" ").trim().to_string()
}

fn strip_ansi_escapes(s: &str) -> String {
    let mut res = String::new();
    let mut in_escape = false;
    for c in s.chars() {
        if c == '\x1b' {
            in_escape = true;
        } else if in_escape {
            if c == 'm' {
                in_escape = false;
            }
        } else {
            res.push(c);
        }
    }
    res
}

#[allow(clippy::collapsible_if)]
fn parse_prefill_tps(stdout: &str) -> f64 {
    for line in stdout.lines() {
        if line.contains("TTFT") && line.contains("tok/s") {
            if let Some(start) = line.find('(') {
                if let Some(end) = line.find(" tok/s") {
                    if start + 1 < end {
                        if let Ok(v) = line[start + 1..end].trim().parse::<f64>() {
                            return v;
                        }
                    }
                }
            }
        }
    }
    0.0
}

fn parse_decode_tps(stdout: &str) -> f64 {
    for line in stdout.lines() {
        if line.contains("Decode Speed") && line.contains("tokens/sec") {
            let parts: Vec<&str> = line.split(':').collect();
            if parts.len() >= 2 {
                let tok_part = parts[1].split("tokens/sec").next().unwrap_or("0").trim();
                if let Ok(v) = tok_part.parse::<f64>() {
                    return v;
                }
            }
        }
    }
    0.0
}
