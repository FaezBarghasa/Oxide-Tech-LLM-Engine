#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // Forward directly to oxide-cli logic
    std::process::Command::new("oxide-engine")
        .args(std::env::args().skip(1))
        .status()?;
    Ok(())
}
