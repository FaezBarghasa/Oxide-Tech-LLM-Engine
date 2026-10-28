use clap::Parser;
use std::path::PathBuf;
use std::process::Command;

#[derive(Parser, Debug)]
#[command(name = "bench-matrix", about = "Empirical Benchmark Matrix: Oxide vs llama.cpp")]
struct Args {
    #[arg(short, long, default_value = "/home/jrad/models/Huihui-Qwen3-8B-abliterated-v2.Q4_K_M.gguf")]
    model: PathBuf,

    #[arg(long, default_value = "16,64,512")]
    prompts: String,

    #[arg(long, default_value_t = 128)]
    gen_tokens: usize,

    #[arg(long, default_value = "target/bench_matrix.json")]
    report: PathBuf,
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Default)]
struct BenchCell {
    test: String,
    llamacpp_tps: f64,
    oxide_tps: f64,
    speedup_ratio: f64,
    passed: bool,
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Default)]
struct BenchReport {
    model: String,
    device: String,
    cells: Vec<BenchCell>,
}

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let args = Args::parse();
    println!("==================================================");
    println!("  OXIDE vs LLAMA.CPP EMPIRICAL BENCHMARK MATRIX   ");
    println!("==================================================");
    println!("Model       : {}", args.model.display());
    println!("Prompt Lens : {}", args.prompts);
    println!("Gen Tokens  : {}", args.gen_tokens);
    println!("Report Path : {}", args.report.display());
    println!("==================================================");

    let mut report = BenchReport {
        model: args.model.to_string_lossy().to_string(),
        device: "NVIDIA GeForce RTX 4060 Laptop dGPU".to_string(),
        cells: Vec::new(),
    };

    // 1. Run llama-bench
    println!("\n[1/2] Invoking llama-bench judge...");
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

    let mut llama_scores = std::collections::HashMap::new();
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

    // 2. Run Oxide on each cell
    println!("\n[2/2] Invoking Oxide-Engine benchmark runs...");
    let prompt_lens: Vec<usize> = args
        .prompts
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();

    for p in &prompt_lens {
        let test_name = format!("pp{p}");
        let llama_tps = *llama_scores.get(&test_name).unwrap_or(&0.0);

        // Generate synthetic prompt of length p
        let words = vec!["word"; *p];
        let p_str = words.join(" ");

        let out = Command::new("target/release/oxide-engine")
            .args([
                "-m",
                &args.model.to_string_lossy(),
                "-b",
                "cuda",
                "-p",
                &p_str,
                "-n",
                "1",
                "--greedy",
                "--speculative",
                "none",
            ])
            .output()?;

        let stdout = String::from_utf8_lossy(&out.stdout);
        let oxide_tps = parse_prefill_tps(&stdout);
        let ratio = if llama_tps > 0.0 { oxide_tps / llama_tps } else { 0.0 };

        report.cells.push(BenchCell {
            test: test_name,
            llamacpp_tps: llama_tps,
            oxide_tps,
            speedup_ratio: ratio,
            passed: ratio >= 1.0,
        });
    }

    // AR Decode test tg128
    let test_name = format!("tg{}", args.gen_tokens);
    let llama_tps = *llama_scores.get(&test_name).unwrap_or(&0.0);
    let out = Command::new("target/release/oxide-engine")
        .args([
            "-m",
            &args.model.to_string_lossy(),
            "-b",
            "cuda",
            "-p",
            "Count to 100:",
            "-n",
            &args.gen_tokens.to_string(),
            "--greedy",
            "--speculative",
            "none",
        ])
        .output()?;
    let stdout = String::from_utf8_lossy(&out.stdout);
    let oxide_tps = parse_decode_tps(&stdout);
    let ratio = if llama_tps > 0.0 { oxide_tps / llama_tps } else { 0.0 };

    report.cells.push(BenchCell {
        test: test_name,
        llamacpp_tps: llama_tps,
        oxide_tps,
        speedup_ratio: ratio,
        passed: ratio >= 0.95,
    });

    // Save JSON
    if let Some(parent) = args.report.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let json_str = serde_json::to_string_pretty(&report)?;
    std::fs::write(&args.report, json_str)?;

    println!("\n==================================================");
    println!("               COMPARISON SCOREBOARD              ");
    println!("==================================================");
    println!("| Test | llama.cpp (t/s) | Oxide (t/s) | Ratio (Oxide / Llama) | Pass/Fail |");
    println!("| :--- | :---: | :---: | :---: | :---: |");
    for cell in &report.cells {
        let status = if cell.passed { "PASS" } else { "FAIL" };
        println!(
            "| {:<6} | {:>15.2} | {:>11.2} | {:>21.2}x | {:^9} |",
            cell.test, cell.llamacpp_tps, cell.oxide_tps, cell.speedup_ratio, status
        );
    }
    println!("==================================================");

    Ok(())
}

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
