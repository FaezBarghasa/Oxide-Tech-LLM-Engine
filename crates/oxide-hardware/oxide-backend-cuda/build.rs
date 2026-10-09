use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn find_nvcc() -> Option<PathBuf> {
    if let Ok(cuda_home) = env::var("CUDA_HOME") {
        let p = Path::new(&cuda_home).join("bin").join("nvcc");
        if p.exists() {
            return Some(p);
        }
    }
    if let Ok(cuda_path) = env::var("CUDA_PATH") {
        let p = Path::new(&cuda_path).join("bin").join("nvcc");
        if p.exists() {
            return Some(p);
        }
    }
    // Check standard locations
    for p in &[
        "/usr/local/cuda-13.2/bin/nvcc",
        "/usr/local/cuda-13/bin/nvcc",
        "/usr/local/cuda-12/bin/nvcc",
        "/usr/local/cuda/bin/nvcc",
        "/usr/bin/nvcc",
    ] {
        let path = Path::new(p);
        if path.exists() {
            return Some(path.to_path_buf());
        }
    }
    None
}

fn main() {
    println!("cargo:rerun-if-changed=src/kernels/llm_kernels.cu");
    println!("cargo:rerun-if-changed=src/kernels/fwht.cu");

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    if let Some(nvcc) = find_nvcc() {
        println!("cargo:rustc-cfg=has_cuda");

        let llm_obj = out_dir.join("llm_kernels.o");
        let fwht_obj = out_dir.join("fwht.o");
        let lib_out = out_dir.join("liboxide_cuda_kernels.a");

        let status = Command::new(&nvcc)
            .args([
                "-O3",
                "-c",
                "-Xcompiler",
                "-fPIC",
                "src/kernels/llm_kernels.cu",
                "-o",
            ])
            .arg(&llm_obj)
            .status();

        if let Ok(s) = status
            && s.success()
        {
            let _ = Command::new(&nvcc)
                .args([
                    "-O3",
                    "-c",
                    "-Xcompiler",
                    "-fPIC",
                    "src/kernels/fwht.cu",
                    "-o",
                ])
                .arg(&fwht_obj)
                .status();

            let ar_status = Command::new("ar")
                .args(["rcs"])
                .arg(&lib_out)
                .arg(&llm_obj)
                .arg(&fwht_obj)
                .status();

            if let Ok(ar_s) = ar_status
                && ar_s.success()
            {
                println!("cargo:rustc-link-search=native={}", out_dir.display());
                println!("cargo:rustc-link-lib=static=oxide_cuda_kernels");
                println!("cargo:rustc-link-lib=dylib=cudart");
                println!("cargo:rustc-link-lib=dylib=cuda");
                println!("cargo:rustc-link-lib=dylib=stdc++");
                return;
            }
        }
    }

    println!(
        "cargo:warning=NVCC not found or compilation skipped; CUDA kernels will run via fallback."
    );
}
