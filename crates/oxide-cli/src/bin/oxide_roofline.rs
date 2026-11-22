//! Hardware Auto-Profiler & Roofline Model (`oxide roofline`)
//!
//! Measures physical hardware sequential memory bandwidth and compute peak,
//! printing a validated Roofline analysis report in the terminal.

use clap::Parser;

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let args = oxide_cli::RooflineArgs::parse();
    oxide_cli::run_roofline_command(&args)
}
