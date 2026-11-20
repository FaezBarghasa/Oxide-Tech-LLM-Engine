//! Oxide Kernel IR (OKIR) — Unified Cross-Target Kernel Intermediate Representation.
//!
//! Provides a unified typed SSA intermediate representation for neural activations, norms,
//! and elementwise operations that lower directly to:
//! - NVIDIA CUDA PTX / C++ (`emit_cuda`)
//! - AMD ROCm HIP (`emit_hip`) with CDNA-64 / RDNA-32 wavefront awareness
//! - Apple Metal Shading Language (`emit_msl`)
//! - Intel Level Zero SPIR-V / OpenCL C (`emit_spirv`)
//! - Vectorized CPU SIMD (`emit_cpu`)

pub mod emit_cpu;
pub mod emit_cuda;
pub mod emit_hip;
pub mod emit_msl;
pub mod emit_spirv;
pub mod ir;

pub use ir::{OkirDataType, OkirFunction, OkirInstruction, OkirModule, OkirOpcode};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(clippy::similar_names)]
    fn test_okir_swiglu_emission_across_all_targets() {
        let func = OkirFunction {
            name: "fused_swiglu".to_string(),
            instructions: vec![
                OkirInstruction {
                    dest_reg: 2,
                    opcode: OkirOpcode::Silu,
                    data_type: OkirDataType::F32,
                    operands: vec![0],
                    immediate: None,
                },
                OkirInstruction {
                    dest_reg: 3,
                    opcode: OkirOpcode::Mul,
                    data_type: OkirDataType::F32,
                    operands: vec![2, 1],
                    immediate: None,
                },
            ],
            input_buffers: vec![
                ("gate".to_string(), OkirDataType::F32),
                ("up".to_string(), OkirDataType::F32),
            ],
            output_buffers: vec![("out".to_string(), OkirDataType::F32)],
            grid_dim: (256, 1, 1),
            block_dim: (256, 1, 1),
        };

        // 1. Emit CUDA
        let cuda_code = emit_cuda::emit_cuda_kernel(&func);
        assert!(cuda_code.contains("__global__ void fused_swiglu_cuda"));
        assert!(cuda_code.contains("__expf(-r0)"));
        assert!(cuda_code.contains("float r3 = r2 * r1;"));

        // 2. Emit HIP (CDNA & RDNA)
        let hip_cdna = emit_hip::emit_hip_kernel(&func, true);
        assert!(hip_cdna.contains("OXIDE_WAVEFRONT_SIZE 64"));
        assert!(hip_cdna.contains("__global__ void fused_swiglu_hip"));

        let hip_rdna = emit_hip::emit_hip_kernel(&func, false);
        assert!(hip_rdna.contains("OXIDE_WAVEFRONT_SIZE 32"));

        // 3. Emit MSL (Metal)
        let msl_code = emit_msl::emit_msl_kernel(&func);
        assert!(msl_code.contains("kernel void fused_swiglu_msl"));
        assert!(msl_code.contains("metal::exp(-r0)"));
        assert!(msl_code.contains("[[buffer(0)]]"));

        // 4. Emit SPIR-V / OpenCL C (Intel)
        let spirv_code = emit_spirv::emit_spirv_opencl_kernel(&func);
        assert!(spirv_code.contains("__kernel void fused_swiglu"));
        assert!(spirv_code.contains("native_exp(-r0)"));

        // 5. Emit CPU Rust
        let cpu_code = emit_cpu::emit_cpu_rust(&func);
        assert!(cpu_code.contains("pub fn fused_swiglu_cpu"));
        assert!(cpu_code.contains("let r2 = r0 / (1.0f32 + (-r0).exp());"));
    }
}
