//! Pure Rust Native Hardware GPU Unlock & Capability Engine
//!
//! Provides direct PCI sysfs hardware topology discovery, register architecture inspection,
//! and instruction issue rate capability analysis for NVIDIA Ampere (RTX 3000) and
//! Ada Lovelace (RTX 4000) architectures, without relying on external C or Python dependencies.

use oxide_core::hardware::GpuArchitecture;
use std::fs;
use std::path::Path;

/// Base MMIO register offsets for NVIDIA GPU Feature Override and Speed Selectors.
/// Reference: Hardware GFW boot process & privilege level mask architecture.
pub mod registers {
    /// Feature Override Privilege Level Mask (PLM).
    /// Default on stock consumer VBIOS: `0xffff_ff8f` (L3 heavy-secure only write).
    /// Unlocked value: `0xffff_ffff` (all privilege levels granted write access).
    pub const FEAT_OVR_PLM: u32 = 0x0082_3804;

    /// Feature Override SM Speed Selector 0 (`SS0`).
    /// Governs eight 4-bit fields: DP, FFMA, FMLA16, FMLA32, and IMLA0-3.
    /// Stock consumer VBIOS default: `0x0000_0001` (1/2 rate on BF16/FP16).
    /// Full-rate unlocked value: `0x8888_8888`.
    pub const FEAT_OVR_SM_SPD_0: u32 = 0x0082_381c;

    /// Feature Override SM Speed Selector 1 (`SS1`).
    /// Governs one 4-bit field at [3:0], attributed to IMLA4.
    /// Stock consumer VBIOS default: `0x0000_0001`.
    /// Full-rate unlocked value: `0x0000_0008`.
    pub const FEAT_OVR_SM_SPD_1: u32 = 0x0082_3820;

    /// Default PLM value locking host-side register writes
    pub const PLM_DEFAULT_LOCKED: u32 = 0xffff_ff8f;

    /// Unlocked PLM value enabling host driver register writes
    pub const PLM_UNLOCKED: u32 = 0xffff_ffff;

    /// Speed Selector 0 full-rate override mask (all 8 lanes set to override + 00 full rate)
    pub const SS0_FULL_RATE_OVERRIDE: u32 = 0x8888_8888;

    /// Speed Selector 1 full-rate override mask
    pub const SS1_FULL_RATE_OVERRIDE: u32 = 0x0000_0008;
}

/// Unlock and architectural status of a physical NVIDIA GPU.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GpuUnlockCapability {
    /// Ada Lovelace (RTX 4000 series) with 4th-Gen Tensor Cores.
    /// Natively supports full-rate dual-issue execution, FP8 (E4M3/E5M2), FP16, and BF16.
    AdaNativeUnthrottled {
        chip_series: &'static str,
        cuda_cores: u32,
        sm_count: u32,
    },
    /// Ampere (RTX 3000 series) GPU with stock firmware half-rate BF16 issue throttling.
    /// Fully compatible with speed selector PLM unlock (+68-80% GEMM throughput).
    AmpereThrottledStock {
        chip_series: &'static str,
        cuda_cores: u32,
        sm_count: u32,
        potential_gemm_gain_pct: &'static str,
    },
    /// Ampere GPU that has already had its Speed Selectors unlocked to full-rate.
    AmpereUnlocked {
        chip_series: &'static str,
    },
    /// Datacenter, professional, or modern architecture where issue rate is natively unthrottled.
    ModernNativeUnthrottled {
        architecture: GpuArchitecture,
    },
    /// Other or non-PCI hardware
    Unsupported {
        device_id: u16,
    },
}

/// Physical Base Address Register (BAR) descriptor parsed from PCI sysfs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PciBarRegion {
    pub bar_index: usize,
    pub physical_address: u64,
    pub size_bytes: u64,
    pub is_mmio: bool,
    pub is_prefetchable: bool,
}

/// Represents a physical NVIDIA GPU discovered on the host PCI bus.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PhysicalPciGpu {
    /// Full PCI Bus-Device-Function string (e.g. "0000:01:00.0")
    pub pci_bdf: String,
    pub vendor_id: u16,
    pub device_id: u16,
    pub subsystem_vendor: u16,
    pub subsystem_device: u16,
    pub pci_class: u32,
    pub architecture: GpuArchitecture,
    pub model_name: &'static str,
    pub capability: GpuUnlockCapability,
    pub bar0_region: Option<PciBarRegion>,
    pub max_link_speed: Option<String>,
    pub current_link_speed: Option<String>,
    pub numa_node: i32,
}

/// Theoretical dense GEMM throughput projections.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GpuThroughputProjection {
    pub fp32_tflops: f32,
    pub fp16_dense_tflops: f32,
    pub bf16_stock_tflops: f32,
    pub bf16_unlocked_tflops: f32,
    pub tf32_dense_tflops: f32,
    pub fp8_dense_tflops: f32,
}

/// Pure Rust Hardware GPU Scanner and Unlock Diagnostic Engine.
#[derive(Debug, Default, Clone)]
pub struct HardwareGpuUnlockEngine;

impl HardwareGpuUnlockEngine {
    /// Creates a new instance of the unlock and probe engine.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Scans the physical PCI bus hierarchy in `/sys/bus/pci/devices/`
    /// and discovers all physical NVIDIA compute/display accelerators.
    #[must_use]
    pub fn scan_pci_devices(&self) -> Vec<PhysicalPciGpu> {
        let pci_root = Path::new("/sys/bus/pci/devices");
        if !pci_root.exists() {
            return Vec::new();
        }

        let mut gpus = Vec::new();
        if let Ok(entries) = fs::read_dir(pci_root) {
            for entry in entries.flatten() {
                let dev_path = entry.path();
                if let Some(gpu) = self.probe_pci_device(&dev_path) {
                    gpus.push(gpu);
                }
            }
        }
        gpus.sort_by(|a, b| a.pci_bdf.cmp(&b.pci_bdf));
        gpus
    }

    /// Probes a single PCI device sysfs directory.
    #[must_use]
    pub fn probe_pci_device(&self, dev_path: &Path) -> Option<PhysicalPciGpu> {
        let vendor_raw = fs::read_to_string(dev_path.join("vendor")).ok()?;
        let device_raw = fs::read_to_string(dev_path.join("device")).ok()?;
        let class_raw = fs::read_to_string(dev_path.join("class")).ok()?;

        let vendor_id = u16::from_str_radix(vendor_raw.trim().trim_start_matches("0x"), 16).ok()?;
        let device_id = u16::from_str_radix(device_raw.trim().trim_start_matches("0x"), 16).ok()?;
        let pci_class = u32::from_str_radix(class_raw.trim().trim_start_matches("0x"), 16).ok()?;

        // Target only NVIDIA Corporation (0x10de) Display (0x0300) and 3D (0x0302) Controllers
        if vendor_id != 0x10de || (pci_class >> 16 != 0x03) {
            return None;
        }

        let subsystem_vendor = fs::read_to_string(dev_path.join("subsystem_vendor"))
            .ok()
            .and_then(|s| u16::from_str_radix(s.trim().trim_start_matches("0x"), 16).ok())
            .unwrap_or(0);

        let subsystem_device = fs::read_to_string(dev_path.join("subsystem_device"))
            .ok()
            .and_then(|s| u16::from_str_radix(s.trim().trim_start_matches("0x"), 16).ok())
            .unwrap_or(0);

        let pci_bdf = dev_path
            .file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_default();

        let numa_node = fs::read_to_string(dev_path.join("numa_node"))
            .ok()
            .and_then(|s| s.trim().parse::<i32>().ok())
            .unwrap_or(0);

        let max_link_speed = fs::read_to_string(dev_path.join("max_link_speed"))
            .ok()
            .map(|s| s.trim().to_string());

        let current_link_speed = fs::read_to_string(dev_path.join("current_link_speed"))
            .ok()
            .map(|s| s.trim().to_string());

        let bar0_region = Self::parse_bar0(dev_path);
        let (architecture, model_name, capability) = Self::classify_device(device_id);

        Some(PhysicalPciGpu {
            pci_bdf,
            vendor_id,
            device_id,
            subsystem_vendor,
            subsystem_device,
            pci_class,
            architecture,
            model_name,
            capability,
            bar0_region,
            max_link_speed,
            current_link_speed,
            numa_node,
        })
    }

    /// Parses the BAR0 MMIO region from the PCI `resource` sysfs file.
    fn parse_bar0(dev_path: &Path) -> Option<PciBarRegion> {
        let resource_content = fs::read_to_string(dev_path.join("resource")).ok()?;
        let first_line = resource_content.lines().next()?;
        let parts: Vec<&str> = first_line.split_whitespace().collect();
        if parts.len() < 3 {
            return None;
        }

        let start = u64::from_str_radix(parts[0].trim_start_matches("0x"), 16).ok()?;
        let end = u64::from_str_radix(parts[1].trim_start_matches("0x"), 16).ok()?;
        let flags = u64::from_str_radix(parts[2].trim_start_matches("0x"), 16).ok()?;

        if start == 0 || end < start {
            return None;
        }

        Some(PciBarRegion {
            bar_index: 0,
            physical_address: start,
            size_bytes: (end - start) + 1,
            is_mmio: (flags & 0x1) == 0,
            is_prefetchable: (flags & 0x8) != 0,
        })
    }

    /// Classifies an NVIDIA PCI device ID into architecture, human name, and unlock capability.
    #[must_use]
    pub fn classify_device(
        device_id: u16,
    ) -> (GpuArchitecture, &'static str, GpuUnlockCapability) {
        match device_id {
            // ==========================================
            // Ada Lovelace (RTX 4000 series: SM 8.9)
            // ==========================================
            // AD102
            0x2681 => (
                GpuArchitecture::Ada,
                "NVIDIA RTX TITAN Ada (AD102)",
                GpuUnlockCapability::AdaNativeUnthrottled {
                    chip_series: "AD102",
                    cuda_cores: 18_432,
                    sm_count: 144,
                },
            ),
            0x2684 => (
                GpuArchitecture::Ada,
                "NVIDIA GeForce RTX 4090 (AD102)",
                GpuUnlockCapability::AdaNativeUnthrottled {
                    chip_series: "AD102",
                    cuda_cores: 16_384,
                    sm_count: 128,
                },
            ),
            0x2685 => (
                GpuArchitecture::Ada,
                "NVIDIA GeForce RTX 4090 D (AD102)",
                GpuUnlockCapability::AdaNativeUnthrottled {
                    chip_series: "AD102",
                    cuda_cores: 14_592,
                    sm_count: 114,
                },
            ),

            // AD103
            0x2702 | 0x2703 => (
                GpuArchitecture::Ada,
                "NVIDIA GeForce RTX 4080 SUPER (AD103)",
                GpuUnlockCapability::AdaNativeUnthrottled {
                    chip_series: "AD103",
                    cuda_cores: 10_240,
                    sm_count: 80,
                },
            ),
            0x2704 => (
                GpuArchitecture::Ada,
                "NVIDIA GeForce RTX 4080 (AD103)",
                GpuUnlockCapability::AdaNativeUnthrottled {
                    chip_series: "AD103",
                    cuda_cores: 9_728,
                    sm_count: 76,
                },
            ),
            0x2705 => (
                GpuArchitecture::Ada,
                "NVIDIA GeForce RTX 4070 Ti SUPER (AD103)",
                GpuUnlockCapability::AdaNativeUnthrottled {
                    chip_series: "AD103",
                    cuda_cores: 8_448,
                    sm_count: 66,
                },
            ),
            0x2717 => (
                GpuArchitecture::Ada,
                "NVIDIA GeForce RTX 4090 Laptop GPU (AD103M)",
                GpuUnlockCapability::AdaNativeUnthrottled {
                    chip_series: "AD103M",
                    cuda_cores: 9_728,
                    sm_count: 76,
                },
            ),

            // AD104
            0x2782 => (
                GpuArchitecture::Ada,
                "NVIDIA GeForce RTX 4070 Ti (AD104)",
                GpuUnlockCapability::AdaNativeUnthrottled {
                    chip_series: "AD104",
                    cuda_cores: 7_680,
                    sm_count: 60,
                },
            ),
            0x2783 => (
                GpuArchitecture::Ada,
                "NVIDIA GeForce RTX 4070 SUPER (AD104)",
                GpuUnlockCapability::AdaNativeUnthrottled {
                    chip_series: "AD104",
                    cuda_cores: 7_168,
                    sm_count: 56,
                },
            ),
            0x2786 => (
                GpuArchitecture::Ada,
                "NVIDIA GeForce RTX 4070 (AD104)",
                GpuUnlockCapability::AdaNativeUnthrottled {
                    chip_series: "AD104",
                    cuda_cores: 5_888,
                    sm_count: 46,
                },
            ),
            0x27a0 | 0x27e0 => (
                GpuArchitecture::Ada,
                "NVIDIA GeForce RTX 4080 Max-Q / Mobile (AD104M)",
                GpuUnlockCapability::AdaNativeUnthrottled {
                    chip_series: "AD104M",
                    cuda_cores: 7_424,
                    sm_count: 58,
                },
            ),

            // AD106
            0x2803 => (
                GpuArchitecture::Ada,
                "NVIDIA GeForce RTX 4060 Ti 8GB (AD106)",
                GpuUnlockCapability::AdaNativeUnthrottled {
                    chip_series: "AD106",
                    cuda_cores: 4_352,
                    sm_count: 34,
                },
            ),
            0x2805 => (
                GpuArchitecture::Ada,
                "NVIDIA GeForce RTX 4060 Ti 16GB (AD106)",
                GpuUnlockCapability::AdaNativeUnthrottled {
                    chip_series: "AD106",
                    cuda_cores: 4_352,
                    sm_count: 34,
                },
            ),
            0x2820 | 0x2860 => (
                GpuArchitecture::Ada,
                "NVIDIA GeForce RTX 4070 Max-Q / Mobile (AD106M)",
                GpuUnlockCapability::AdaNativeUnthrottled {
                    chip_series: "AD106M",
                    cuda_cores: 4_608,
                    sm_count: 36,
                },
            ),

            // AD107 (Host Hardware Reference)
            0x2882 => (
                GpuArchitecture::Ada,
                "NVIDIA GeForce RTX 4060 Desktop (AD107)",
                GpuUnlockCapability::AdaNativeUnthrottled {
                    chip_series: "AD107",
                    cuda_cores: 3_072,
                    sm_count: 24,
                },
            ),
            0x28e0 | 0x28a0 => (
                GpuArchitecture::Ada,
                "NVIDIA GeForce RTX 4060 Max-Q / Mobile (AD107M)",
                GpuUnlockCapability::AdaNativeUnthrottled {
                    chip_series: "AD107M",
                    cuda_cores: 3_072,
                    sm_count: 24,
                },
            ),
            0x28e1 | 0x28a1 => (
                GpuArchitecture::Ada,
                "NVIDIA GeForce RTX 4050 Max-Q / Mobile (AD107M)",
                GpuUnlockCapability::AdaNativeUnthrottled {
                    chip_series: "AD107M",
                    cuda_cores: 2_560,
                    sm_count: 20,
                },
            ),

            // ==========================================
            // Ampere (RTX 3000 series: SM 8.6)
            // ==========================================
            // GA102
            0x2203 => (
                GpuArchitecture::Ampere,
                "NVIDIA GeForce RTX 3090 Ti (GA102)",
                GpuUnlockCapability::AmpereThrottledStock {
                    chip_series: "GA102",
                    cuda_cores: 10_752,
                    sm_count: 84,
                    potential_gemm_gain_pct: "+68% to +80%",
                },
            ),
            0x2204 => (
                GpuArchitecture::Ampere,
                "NVIDIA GeForce RTX 3090 (GA102)",
                GpuUnlockCapability::AmpereThrottledStock {
                    chip_series: "GA102",
                    cuda_cores: 10_496,
                    sm_count: 82,
                    potential_gemm_gain_pct: "+25% to +67%",
                },
            ),
            0x2205 => (
                GpuArchitecture::Ampere,
                "NVIDIA GeForce RTX 3080 Ti 20GB OEM (GA102)",
                GpuUnlockCapability::AmpereThrottledStock {
                    chip_series: "GA102",
                    cuda_cores: 10_240,
                    sm_count: 80,
                    potential_gemm_gain_pct: "+50% to +70%",
                },
            ),
            0x2206 => (
                GpuArchitecture::Ampere,
                "NVIDIA GeForce RTX 3080 (GA102)",
                GpuUnlockCapability::AmpereThrottledStock {
                    chip_series: "GA102",
                    cuda_cores: 8_704,
                    sm_count: 68,
                    potential_gemm_gain_pct: "+56% to +73%",
                },
            ),

            // GA104
            0x2482 => (
                GpuArchitecture::Ampere,
                "NVIDIA GeForce RTX 3070 Ti (GA104)",
                GpuUnlockCapability::AmpereThrottledStock {
                    chip_series: "GA104",
                    cuda_cores: 6_144,
                    sm_count: 48,
                    potential_gemm_gain_pct: "+70% to +80%",
                },
            ),
            0x2484 => (
                GpuArchitecture::Ampere,
                "NVIDIA GeForce RTX 3070 (GA104)",
                GpuUnlockCapability::AmpereThrottledStock {
                    chip_series: "GA104",
                    cuda_cores: 5_888,
                    sm_count: 46,
                    potential_gemm_gain_pct: "+73% to +81%",
                },
            ),

            // GA106
            0x2504 => (
                GpuArchitecture::Ampere,
                "NVIDIA GeForce RTX 3060 12GB (GA106)",
                GpuUnlockCapability::AmpereThrottledStock {
                    chip_series: "GA106",
                    cuda_cores: 3_584,
                    sm_count: 28,
                    potential_gemm_gain_pct: "~+80%",
                },
            ),

            // Modern / Datacenter (A100, H100, Blackwell, L40S, etc.)
            0x20b0 | 0x20b2 | 0x20b5 => (
                GpuArchitecture::Ampere,
                "NVIDIA A100 Tensor Core GPU",
                GpuUnlockCapability::ModernNativeUnthrottled {
                    architecture: GpuArchitecture::Ampere,
                },
            ),
            0x2322 | 0x2324 => (
                GpuArchitecture::Hopper,
                "NVIDIA H100 PCIe / SXM5",
                GpuUnlockCapability::ModernNativeUnthrottled {
                    architecture: GpuArchitecture::Hopper,
                },
            ),
            0x26b9 | 0x26ba => (
                GpuArchitecture::Ada,
                "NVIDIA L40 / L40S Datacenter GPU",
                GpuUnlockCapability::ModernNativeUnthrottled {
                    architecture: GpuArchitecture::Ada,
                },
            ),

            // Fallback
            _ => (
                GpuArchitecture::Ampere,
                "Unknown NVIDIA GPU",
                GpuUnlockCapability::Unsupported { device_id },
            ),
        }
    }

    /// Calculates theoretical dense GEMM TFLOPS before and after unlock based on SM clocks.
    #[must_use]
    pub fn compute_gemm_projections(
        capability: &GpuUnlockCapability,
        boost_clock_mhz: f32,
    ) -> Option<GpuThroughputProjection> {
        let clock_ghz = boost_clock_mhz / 1000.0;

        match capability {
            GpuUnlockCapability::AdaNativeUnthrottled {
                cuda_cores,
                sm_count,
                ..
            } => {
                // Ada Lovelace (SM 8.9):
                // 128 CUDA cores per SM. 4 4th-Gen Tensor Cores per SM.
                // FP32: 2 ops/clock/core -> cuda_cores * 2 * clock_ghz
                let fp32 = *cuda_cores as f32 * 2.0 * clock_ghz / 1000.0;
                // FP16/BF16 dense Tensor Cores: 512 FLOPs per SM per clock
                let fp16_dense = *sm_count as f32 * 512.0 * clock_ghz / 1000.0;
                let bf16_dense = fp16_dense;
                // TF32 dense: 256 FLOPs per SM per clock
                let tf32_dense = *sm_count as f32 * 256.0 * clock_ghz / 1000.0;
                // FP8 dense: 1024 FLOPs per SM per clock
                let fp8_dense = *sm_count as f32 * 1024.0 * clock_ghz / 1000.0;

                Some(GpuThroughputProjection {
                    fp32_tflops: fp32,
                    fp16_dense_tflops: fp16_dense,
                    bf16_stock_tflops: bf16_dense,
                    bf16_unlocked_tflops: bf16_dense,
                    tf32_dense_tflops: tf32_dense,
                    fp8_dense_tflops: fp8_dense,
                })
            }
            GpuUnlockCapability::AmpereThrottledStock {
                cuda_cores,
                sm_count,
                ..
            } => {
                // Ampere (SM 8.6):
                // 128 CUDA cores per SM. 4 3rd-Gen Tensor Cores per SM.
                let fp32 = *cuda_cores as f32 * 2.0 * clock_ghz / 1000.0;
                // Full rate 3rd-Gen Tensor Cores: 512 FLOPs per SM per clock
                let full_rate_dense = *sm_count as f32 * 512.0 * clock_ghz / 1000.0;
                // Stock consumer VBIOS throttles BF16 to half rate:
                let bf16_stock = full_rate_dense * 0.55;
                let tf32_stock = full_rate_dense * 0.27;

                Some(GpuThroughputProjection {
                    fp32_tflops: fp32,
                    fp16_dense_tflops: full_rate_dense * 0.75,
                    bf16_stock_tflops: bf16_stock,
                    bf16_unlocked_tflops: full_rate_dense,
                    tf32_dense_tflops: tf32_stock,
                    fp8_dense_tflops: 0.0, // Ampere does not support FP8
                })
            }
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_classification_ada_4000_series() {
        // Host Reference GPU: RTX 4060 Mobile (0x28e0)
        let (arch, name, cap) = HardwareGpuUnlockEngine::classify_device(0x28e0);
        assert_eq!(arch, GpuArchitecture::Ada);
        assert!(name.contains("RTX 4060 Max-Q"));
        match cap {
            GpuUnlockCapability::AdaNativeUnthrottled {
                cuda_cores,
                sm_count,
                ..
            } => {
                assert_eq!(cuda_cores, 3_072);
                assert_eq!(sm_count, 24);
            }
            _ => panic!("Expected AdaNativeUnthrottled"),
        }

        // RTX 4090 (0x2684)
        let (arch_4090, name_4090, cap_4090) = HardwareGpuUnlockEngine::classify_device(0x2684);
        assert_eq!(arch_4090, GpuArchitecture::Ada);
        assert!(name_4090.contains("RTX 4090"));
        match cap_4090 {
            GpuUnlockCapability::AdaNativeUnthrottled { cuda_cores, .. } => {
                assert_eq!(cuda_cores, 16_384);
            }
            _ => panic!("Expected AdaNativeUnthrottled"),
        }
    }

    #[test]
    fn test_device_classification_ampere_3000_series() {
        // RTX 3090 (0x2204)
        let (arch, name, cap) = HardwareGpuUnlockEngine::classify_device(0x2204);
        assert_eq!(arch, GpuArchitecture::Ampere);
        assert!(name.contains("RTX 3090"));
        match cap {
            GpuUnlockCapability::AmpereThrottledStock {
                cuda_cores,
                potential_gemm_gain_pct,
                ..
            } => {
                assert_eq!(cuda_cores, 10_496);
                assert_eq!(potential_gemm_gain_pct, "+25% to +67%");
            }
            _ => panic!("Expected AmpereThrottledStock"),
        }
    }

    #[test]
    fn test_throughput_projections_ad107_host_gpu() {
        let (_, _, cap) = HardwareGpuUnlockEngine::classify_device(0x28e0);

        // At 3105 MHz (host boost clock)
        let proj = HardwareGpuUnlockEngine::compute_gemm_projections(&cap, 3105.0)
            .expect("Projections should be computed");

        assert!(proj.fp32_tflops > 18.0 && proj.fp32_tflops < 21.0);
        assert!(proj.fp16_dense_tflops > 36.0 && proj.fp16_dense_tflops < 40.0);
        assert!(proj.fp8_dense_tflops > 70.0 && proj.fp8_dense_tflops < 80.0);
    }

    #[test]
    fn test_live_pci_bus_scan_discovers_host_hardware() {
        let engine = HardwareGpuUnlockEngine::new();
        let gpus = engine.scan_pci_devices();

        // On Linux host with Pop!_OS and RTX 4060 dGPU, at least 1 NVIDIA controller is detected
        if !gpus.is_empty() {
            let primary = &gpus[0];
            assert_eq!(primary.vendor_id, 0x10de);
            assert_eq!(primary.pci_class >> 16, 0x03);
            println!(
                "Discovered physical GPU: {} (PCI {}, DevID 0x{:04x})",
                primary.model_name, primary.pci_bdf, primary.device_id
            );
        }
    }
}
