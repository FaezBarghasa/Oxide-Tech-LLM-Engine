use oxide_quant::int_quant::{BlockQ8_0, f16};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;

#[derive(Debug, Clone)]
pub struct BenchMetrics {
    pub concurrency: usize,
    pub total_tokens: usize,
    pub elapsed: Duration,
    pub p50_itl_ms: f64,
    pub p99_itl_ms: f64,
    pub tokens_per_second: f64,
}

async fn execute_client_request(
    _client_id: usize,
    matrix: Arc<Vec<BlockQ8_0>>,
    dim_m: usize,
    dim_n: usize,
    gen_len: usize,
    metric_tx: mpsc::Sender<Duration>,
) {
    let mut activation_vector = vec![0.1f32; dim_n];
    let mut output = vec![0.0f32; dim_m];

    // 1. Real Prefill GEMV calculation
    oxide_quant::simd::gemv_q8_0(&matrix, &activation_vector, dim_m, dim_n, &mut output);

    // 2. Real Decode GEMV steps
    for step in 0..gen_len {
        let step_start = Instant::now();
        // Feedback output into activation vector to simulate real autoregressive decode state
        for (i, val) in output.iter().take(dim_n).enumerate() {
            activation_vector[i] = *val * 0.01 + (step as f32 * 0.001);
        }
        oxide_quant::simd::gemv_q8_0(&matrix, &activation_vector, dim_m, dim_n, &mut output);
        let step_latency = step_start.elapsed();
        let _ = metric_tx.send(step_latency).await;
    }
}

pub async fn run_continuous_serving_suite(
    concurrency: usize,
    requests_per_client: usize,
) -> BenchMetrics {
    let (tx, mut rx) = mpsc::channel(100_000);
    let start_all = Instant::now();

    let dim_m = 256;
    let dim_n = 256; // 8 blocks of 32 per row
    let blocks_per_row = dim_n / 32;
    let total_blocks = dim_m * blocks_per_row;
    let mut raw_blocks = vec![BlockQ8_0::default(); total_blocks];
    for (i, blk) in raw_blocks.iter_mut().enumerate() {
        blk.scale = f16::from_f32(0.02);
        for b in 0..32 {
            blk.qs[b] = (((i * 3 + b) % 255) as i32 - 128) as i8;
        }
    }
    let matrix = Arc::new(raw_blocks);

    let mut handles = Vec::new();
    for c in 0..concurrency {
        let tx_clone = tx.clone();
        let mat_clone = Arc::clone(&matrix);
        handles.push(tokio::spawn(async move {
            for _ in 0..requests_per_client {
                execute_client_request(
                    c,
                    Arc::clone(&mat_clone),
                    dim_m,
                    dim_n,
                    32,
                    tx_clone.clone(),
                )
                .await;
            }
        }));
    }
    drop(tx);

    let mut latencies = Vec::new();
    while let Some(lat) = rx.recv().await {
        latencies.push(lat.as_secs_f64() * 1000.0);
    }

    for h in handles {
        h.await.unwrap();
    }

    let elapsed = start_all.elapsed();
    latencies.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let total_tokens = latencies.len();
    let p50 = if total_tokens > 0 {
        latencies[total_tokens * 50 / 100]
    } else {
        0.0
    };
    let p99 = if total_tokens > 0 {
        latencies[total_tokens * 99 / 100]
    } else {
        0.0
    };
    let tps = if elapsed.as_secs_f64() > 0.0 {
        (total_tokens as f64) / elapsed.as_secs_f64()
    } else {
        0.0
    };

    BenchMetrics {
        concurrency,
        total_tokens,
        elapsed,
        p50_itl_ms: p50,
        p99_itl_ms: p99,
        tokens_per_second: tps,
    }
}

#[tokio::main]
async fn main() {
    println!("=== Oxide-Tech-LLM-Engine Continuous Serving Benchmark ===");
    for &concurrency in &[1, 4, 8, 16] {
        let metrics = run_continuous_serving_suite(concurrency, 2).await;
        println!(
            "Concurrency: {:2} | Total Tokens: {:5} | Elapsed: {:6.2?} | P50 ITL: {:.2}ms | P99 ITL: {:.2}ms | Throughput: {:.1} tok/s",
            metrics.concurrency,
            metrics.total_tokens,
            metrics.elapsed,
            metrics.p50_itl_ms,
            metrics.p99_itl_ms,
            metrics.tokens_per_second
        );
    }
}
