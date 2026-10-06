//! `oxide-quantize`: CLI tool for quantizing FP32/FP16 tensors into standard GGUF quant formats (Q4_0, Q8_0, Q4_K).

use clap::Parser;
use oxide_quant::gguf_quants::BlockQ4_K;
use oxide_quant::int_quant::{BlockQ4_0, BlockQ8_0};
use std::fs::File;
use std::io::{Read, Write};
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(
    name = "oxide-quantize",
    about = "Oxide-Tech-LLM-Engine: High-throughput offline quantization tool (FP32 -> Q4_0, Q8_0, Q4_K)"
)]
struct Args {
    /// Input raw binary or tensor file (containing contiguous IEEE-754 FP32 floats)
    #[arg(short, long)]
    input: PathBuf,

    /// Output quantized file destination
    #[arg(short, long)]
    output: PathBuf,

    /// Target quantization format: "q4_0", "q8_0", or "q4_k"
    #[arg(short, long, default_value = "q4_k")]
    format: String,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    println!("Oxide-Quantize | Target Architecture: AVX-512 / AVX2 / AMX accelerated");
    println!("Processing: {:?} -> {:?}", args.input, args.output);
    println!("Selected Quantization Format: {}", args.format);

    let start_time = Instant::now();

    // Read input data
    let mut file = File::open(&args.input)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;

    if buffer.len() % 4 != 0 {
        eprintln!(
            "Warning: Input file size is not a multiple of 4 bytes (f32 alignment). Truncating."
        );
    }
    let f32_count = buffer.len() / 4;
    println!(
        "Loaded {f32_count} input floating point values ({:.2} MB)",
        buffer.len() as f64 / 1_048_576.0
    );

    let mut f32_values = Vec::with_capacity(f32_count);
    for chunk in buffer.as_chunks::<4>().0 {
        f32_values.push(f32::from_le_bytes(*chunk));
    }

    let mut out_file = File::create(&args.output)?;

    match args.format.to_lowercase().as_str() {
        "q4_0" => {
            let blocks_count = f32_count / 32;
            let mut quantized_blocks = Vec::with_capacity(blocks_count);
            for i in 0..blocks_count {
                let chunk: &[f32; 32] = f32_values[i * 32..(i + 1) * 32].try_into()?;
                quantized_blocks.push(BlockQ4_0::quantize(chunk));
            }
            // SAFETY: BlockQ4_0 is repr(C) with no padding pointers
            let bytes_slice = unsafe {
                std::slice::from_raw_parts(
                    quantized_blocks.as_ptr().cast::<u8>(),
                    quantized_blocks.len() * std::mem::size_of::<BlockQ4_0>(),
                )
            };
            out_file.write_all(bytes_slice)?;
            println!(
                "Quantized {} blocks of Q4_0 ({} bytes written)",
                blocks_count,
                bytes_slice.len()
            );
        }
        "q8_0" => {
            let blocks_count = f32_count / 32;
            let mut quantized_blocks = Vec::with_capacity(blocks_count);
            for i in 0..blocks_count {
                let chunk: &[f32; 32] = f32_values[i * 32..(i + 1) * 32].try_into()?;
                quantized_blocks.push(BlockQ8_0::quantize(chunk));
            }
            // SAFETY: BlockQ8_0 is repr(C)
            let bytes_slice = unsafe {
                std::slice::from_raw_parts(
                    quantized_blocks.as_ptr().cast::<u8>(),
                    quantized_blocks.len() * std::mem::size_of::<BlockQ8_0>(),
                )
            };
            out_file.write_all(bytes_slice)?;
            println!(
                "Quantized {} blocks of Q8_0 ({} bytes written)",
                blocks_count,
                bytes_slice.len()
            );
        }
        "q4_k" => {
            let superblocks_count = f32_count / 256;
            let mut quantized_blocks = Vec::with_capacity(superblocks_count);
            for i in 0..superblocks_count {
                let chunk: &[f32; 256] = f32_values[i * 256..(i + 1) * 256].try_into()?;
                quantized_blocks.push(BlockQ4_K::quantize(chunk));
            }
            // SAFETY: BlockQ4_K is repr(C)
            let bytes_slice = unsafe {
                std::slice::from_raw_parts(
                    quantized_blocks.as_ptr().cast::<u8>(),
                    quantized_blocks.len() * std::mem::size_of::<BlockQ4_K>(),
                )
            };
            out_file.write_all(bytes_slice)?;
            println!(
                "Quantized {} superblocks of Q4_K ({} bytes written)",
                superblocks_count,
                bytes_slice.len()
            );
        }
        other => {
            eprintln!("Unsupported format '{other}'. Choose from: q4_0, q8_0, q4_k");
            std::process::exit(1);
        }
    }

    let elapsed = start_time.elapsed();
    println!(
        "Quantization complete in {:.2?} ({:.2} MB/s)",
        elapsed,
        (buffer.len() as f64 / 1_048_576.0) / elapsed.as_secs_f64()
    );
    Ok(())
}
