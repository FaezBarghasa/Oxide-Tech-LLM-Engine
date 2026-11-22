//! Standalone Verification Tool (`oxide verify`)
//!
//! Verifies golden tensor parity, cosine similarity, max absolute error,
//! and deterministic token generation against reference.

use clap::Parser;

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let args = oxide_cli::VerifyArgs::parse();
    oxide_cli::run_verify_command(&args)
}
