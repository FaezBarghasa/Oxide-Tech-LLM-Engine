use std::fmt;

/// GPU / Accelerator Microarchitecture Family.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GpuArchitecture {
    // NVIDIA Architectures
    Volta,     // SM 7.0 (V100)
    Turing,    // SM 7.5 (RTX 2000 series, T4, Quadro RTX)
    Ampere,    // SM 8.0 / 8.6 (RTX 3000 series, A100, A10, A30, RTX A6000)
    Orin,      // SM 8.7 (Jetson AGX Orin, Orin Nano, Orin NX)
    Ada,       // SM 8.9 (RTX 4000 series, RTX 6000 Ada, L40S, L4)
    Hopper,    // SM 9.0 (H100, H200, H800, H20)
    Blackwell, // SM 10.0 / 12.0 (B100, B200, B300, GB200, GB300, RTX 5000 series)

    // AMD CDNA Datacenter / Instinct Architectures
    Cdna1, // gfx908 (MI100)
    Cdna2, // gfx90a (MI200, MI250X, MI210)
    Cdna3, // gfx942 (MI300X 192GB, MI300A APU, MI325X 256GB)
    Cdna4, // gfx950 (MI350X 288GB, MI355X 288GB 3nm FP4/FP6/FP8)

    // AMD RDNA Commercial / Gaming / Workstation Architectures
    Rdna1,   // gfx1010 (Radeon RX 5000 series)
    Rdna2,   // gfx1030 (Radeon RX 6000 series, PRO W6800)
    Rdna3,   // gfx1100 (Radeon RX 7900 XTX, 7900 XT, 7800 XT, PRO W7900)
    Rdna3_5, // gfx1150 (Ryzen AI 300 Strix Point / Strix Halo APU)
    Rdna4,   // gfx1200 (Radeon RX 8000 series)

    // AMD XDNA NPU Architectures
    XdnaNpu, // XDNA 1 / XDNA 2 Tile Engine (10-55 TOPS)

    // Google Cloud & Edge TPU Architectures
    GoogleTpuV2,          // TPU v2 (128x128 MXU, HBM)
    GoogleTpuV3,          // TPU v3 (Dual 128x128 MXU, HBM2, Liquid Cooled)
    GoogleTpuV4,          // TPU v4 / v4i (Quad 128x128 MXU, 3D Torus OCS, HBM2)
    GoogleTpuV5e,         // TPU v5e ViperLite (LLM Inference/Training cost-optimized)
    GoogleTpuV5p,         // TPU v5p (95GB HBM2e, 459 TFLOPS BF16, 4800 Gbps 3D Torus ICI)
    GoogleTpuV6eTrillium, // TPU v6e Trillium (32GB HBM3, 920 TFLOPS BF16/FP8, 3rd Gen SparseCore)
    GoogleEdgeTpu,        // Google Coral Edge TPU (4 TOPS INT8, PCIe/USB/M.2)
}

/// Compute Capability / Target ISA Version.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ComputeCapability {
    pub major: u32,
    pub minor: u32,
}

impl ComputeCapability {
    // NVIDIA
    pub const SM_70_VOLTA: Self = Self { major: 7, minor: 0 };
    pub const SM_75_TURING: Self = Self { major: 7, minor: 5 };
    pub const SM_80_AMPERE_DC: Self = Self { major: 8, minor: 0 };
    pub const SM_86_AMPERE_CLIENT: Self = Self { major: 8, minor: 6 };
    pub const SM_87_ORIN: Self = Self { major: 8, minor: 7 };
    pub const SM_89_ADA: Self = Self { major: 8, minor: 9 };
    pub const SM_90_HOPPER: Self = Self { major: 9, minor: 0 };
    pub const SM_100_BLACKWELL: Self = Self {
        major: 10,
        minor: 0,
    };
    pub const SM_120_BLACKWELL_CLIENT: Self = Self {
        major: 12,
        minor: 0,
    };

    // AMD GFX Targets
    pub const GFX_908_CDNA1: Self = Self { major: 9, minor: 8 };
    pub const GFX_90A_CDNA2: Self = Self {
        major: 9,
        minor: 10,
    };
    pub const GFX_942_CDNA3: Self = Self {
        major: 9,
        minor: 42,
    };
    pub const GFX_950_CDNA4: Self = Self {
        major: 9,
        minor: 50,
    };
    pub const GFX_1030_RDNA2: Self = Self {
        major: 10,
        minor: 30,
    };
    pub const GFX_1100_RDNA3: Self = Self {
        major: 11,
        minor: 0,
    };
    pub const GFX_1150_RDNA3_5: Self = Self {
        major: 11,
        minor: 50,
    };
    pub const GFX_1151_RDNA3_5: Self = Self {
        major: 11,
        minor: 51,
    };
    pub const GFX_1200_RDNA4: Self = Self {
        major: 12,
        minor: 0,
    };

    // Google TPU Targets
    pub const TPU_V2: Self = Self {
        major: 20,
        minor: 2,
    };
    pub const TPU_V3: Self = Self {
        major: 20,
        minor: 3,
    };
    pub const TPU_V4: Self = Self {
        major: 20,
        minor: 4,
    };
    pub const TPU_V5E: Self = Self {
        major: 20,
        minor: 50,
    };
    pub const TPU_V5P: Self = Self {
        major: 20,
        minor: 51,
    };
    pub const TPU_V6E_TRILLIUM: Self = Self {
        major: 20,
        minor: 60,
    };
    pub const EDGE_TPU: Self = Self {
        major: 20,
        minor: 1,
    };

    #[must_use]
    pub const fn new(major: u32, minor: u32) -> Self {
        Self { major, minor }
    }

    #[must_use]
    pub const fn architecture(self) -> GpuArchitecture {
        match (self.major, self.minor) {
            (7, 0) => GpuArchitecture::Volta,
            (7, 5) => GpuArchitecture::Turing,
            (8, 7) => GpuArchitecture::Orin,
            (8, 9) => GpuArchitecture::Ada,
            (9, 0..=7) => GpuArchitecture::Hopper,
            (9, 8) => GpuArchitecture::Cdna1,
            (9, 10) => GpuArchitecture::Cdna2,
            (9, 42) => GpuArchitecture::Cdna3,
            (9, 50) => GpuArchitecture::Cdna4,
            (10, 30) => GpuArchitecture::Rdna2,
            (11, 0) => GpuArchitecture::Rdna3,
            (11, 50..=51) => GpuArchitecture::Rdna3_5,
            (12, 0) => GpuArchitecture::Rdna4,
            (10..=12, _) => GpuArchitecture::Blackwell,
            (20, 1) => GpuArchitecture::GoogleEdgeTpu,
            (20, 2) => GpuArchitecture::GoogleTpuV2,
            (20, 3) => GpuArchitecture::GoogleTpuV3,
            (20, 4) => GpuArchitecture::GoogleTpuV4,
            (20, 50) => GpuArchitecture::GoogleTpuV5e,
            (20, 51) => GpuArchitecture::GoogleTpuV5p,
            (20, 60) => GpuArchitecture::GoogleTpuV6eTrillium,
            _ => GpuArchitecture::Ampere,
        }
    }
}

impl fmt::Display for ComputeCapability {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.major == 20 {
            match self.minor {
                1 => write!(f, "tpu_edge"),
                2 => write!(f, "tpu_v2"),
                3 => write!(f, "tpu_v3"),
                4 => write!(f, "tpu_v4"),
                50 => write!(f, "tpu_v5e"),
                51 => write!(f, "tpu_v5p"),
                60 => write!(f, "tpu_v6e_trillium"),
                _ => write!(f, "tpu_v{}", self.minor),
            }
        } else if self.major == 9
            && (self.minor == 8 || self.minor == 10 || self.minor == 42 || self.minor == 50)
        {
            write!(f, "gfx9{:02x}", self.minor)
        } else if self.major >= 10
            && self.major <= 12
            && (self.minor == 30 || self.minor == 50 || self.minor == 51 || self.minor == 0)
        {
            write!(f, "gfx{}{}", self.major, self.minor)
        } else {
            write!(f, "sm_{}{}", self.major, self.minor)
        }
    }
}

/// Physical and Deployment Hardware Form Factor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HardwareFormFactor {
    DesktopWorkstation, // Standard Desktop PCIe (e.g. N1X Workstations, custom rigs, Radeon RX)
    LaptopMobile,       // High-efficiency Mobile Max-Q / Laptop GPUs / APUs
    EnterpriseRackServer, // 1U-8U Enterprise Server (e.g. N1X Servers)
    DatacenterSxmNvl,   // SXM5 / SXM6 / NVL72 High-Density Multi-GPU
    DatacenterOamInstinct, // OAM / UBB AMD Instinct Datacenter Module
    DatacenterTpuPod3dTorus, // Google Cloud TPU v4/v5p 3D Torus OCS Pod
    DatacenterTpuPod2dTorus, // Google Cloud TPU v2/v3/v5e/v6e 2D Torus Pod
    EdgeEmbedded,       // Jetson Orin / Embedded APU Modules
    EdgeTpuModule,      // Google Coral Edge TPU USB/PCIe/M.2
    DgxStationSpark,    // NVIDIA DGX Spark / Station AI nodes
    SuperchipGraceBlackwell, // Grace-Blackwell Coherent Memory Substrate
    ApuUnifiedMemoryWithNpu, // AMD Ryzen AI / Strix Point / MI300A Coherent Unified APU + XDNA NPU
}

/// Memory silicon technology.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemoryTechnology {
    Gddr6,
    Gddr6X,
    Gddr7,
    Hbm,
    Hbm2,
    Hbm2e,
    Hbm3,
    Hbm3e,
    UnifiedLpddr5X,
    UnifiedDdr5Coherent,
    SramOnChip,
}

/// Tensor / Matrix Core hardware generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TensorCoreGeneration {
    Gen1Volta,               // NVIDIA FP16 MMA
    Gen2Turing,              // NVIDIA INT4, INT8, FP16
    Gen3Ampere,              // NVIDIA BF16, TF32, Structured 2:4 Sparsity
    Gen4HopperAda,           // NVIDIA FP8 E4M3/E5M2, DPX, Async TMA
    Gen5Blackwell,           // NVIDIA NVFP4, Microscaling MXFP4/MXFP8
    AmdMfmaCdna3,            // AMD CDNA 3 MFMA (MI300X/MI325X: FP8, BF16, INT8, FP16)
    AmdMfmaCdna4,            // AMD CDNA 4 MFMA (MI350X/MI355X: FP4, FP6, FP8, Microscaling)
    AmdRdnaWmma,             // AMD RDNA 3/3.5/4 WMMA Matrix Accelerator
    AmdXdnaNpuEngine,        // AMD XDNA 1 / XDNA 2 Spatial NPU Tile Array
    GoogleTpuMxuV2,          // Google TPU v2 128x128 BF16 Matrix Multiply Unit
    GoogleTpuMxuV3,          // Google TPU v3 Dual 128x128 BF16 MXU
    GoogleTpuMxuV4,          // Google TPU v4 Quad 128x128 BF16/INT8 MXU + SparseCore
    GoogleTpuMxuV5,          // Google TPU v5e/v5p MXU + 2nd Gen SparseCore
    GoogleTpuMxuV6Trillium,  // Google TPU v6e Trillium FP8/BF16/INT8 MXU + 3rd Gen SparseCore
    GoogleEdgeTpuInt8Engine, // Google Coral Edge TPU 4 TOPS INT8 Systolic Engine
}

/// Comprehensive hardware profiling descriptor for target GPU / accelerator / APU.
#[derive(Debug, Clone, PartialEq)]
pub struct GpuDeviceProfile {
    pub name: String,
    pub compute_capability: ComputeCapability,
    pub architecture: GpuArchitecture,
    pub form_factor: HardwareFormFactor,
    pub memory_tech: MemoryTechnology,
    pub tensor_core_gen: TensorCoreGeneration,
    pub sm_count: u32,
    pub vram_capacity_bytes: u64,
    pub memory_bus_width_bits: u32,
    pub memory_bandwidth_gbps: f32,
    pub l2_cache_bytes: usize,
    pub smem_per_sm_bytes: usize,
    pub smem_per_block_bytes: usize,
    pub max_threads_per_sm: u32,
    pub supports_tma: bool,
    pub supports_fp8: bool,
    pub supports_nvfp4: bool,
    pub supports_async_copy: bool,
    pub supports_nvlink: bool,
    pub nvlink_bandwidth_gbps: f32,
}

impl GpuDeviceProfile {
    /// Discovers and constructs the profile for a well-known NVIDIA or AMD GPU / APU.
    #[must_use]
    pub fn from_known_device_name(name: &str) -> Option<Self> {
        let n = name.to_lowercase();

        // ==========================================
        // 1. AMD INSTINCT DATACENTER ACCELERATORS
        // ==========================================

        // AMD Instinct MI350X & MI355X (CDNA 4, gfx950, 288GB HBM3e, FP4/FP6/FP8)
        if n.contains("mi350") || n.contains("mi355") {
            return Some(Self {
                name: name.to_string(),
                compute_capability: ComputeCapability::GFX_950_CDNA4,
                architecture: GpuArchitecture::Cdna4,
                form_factor: HardwareFormFactor::DatacenterOamInstinct,
                memory_tech: MemoryTechnology::Hbm3e,
                tensor_core_gen: TensorCoreGeneration::AmdMfmaCdna4,
                sm_count: 320, // Compute Units
                vram_capacity_bytes: 288 * 1024 * 1024 * 1024,
                memory_bus_width_bits: 8192,
                memory_bandwidth_gbps: 8000.0,
                l2_cache_bytes: 256 * 1024 * 1024,
                smem_per_sm_bytes: 128 * 1024,
                smem_per_block_bytes: 64 * 1024,
                max_threads_per_sm: 2048,
                supports_tma: true,
                supports_fp8: true,
                supports_nvfp4: true, // AMD FP4 MFMA
                supports_async_copy: true,
                supports_nvlink: true, // Infinity Fabric 1.2 TB/s
                nvlink_bandwidth_gbps: 1200.0,
            });
        }

        // AMD Instinct MI325X (CDNA 3, gfx942, 256GB HBM3e, 6.0 TB/s)
        if n.contains("mi325") {
            return Some(Self {
                name: name.to_string(),
                compute_capability: ComputeCapability::GFX_942_CDNA3,
                architecture: GpuArchitecture::Cdna3,
                form_factor: HardwareFormFactor::DatacenterOamInstinct,
                memory_tech: MemoryTechnology::Hbm3e,
                tensor_core_gen: TensorCoreGeneration::AmdMfmaCdna3,
                sm_count: 304,
                vram_capacity_bytes: 256 * 1024 * 1024 * 1024,
                memory_bus_width_bits: 8192,
                memory_bandwidth_gbps: 6000.0,
                l2_cache_bytes: 256 * 1024 * 1024,
                smem_per_sm_bytes: 128 * 1024,
                smem_per_block_bytes: 64 * 1024,
                max_threads_per_sm: 2048,
                supports_tma: false,
                supports_fp8: true,
                supports_nvfp4: false,
                supports_async_copy: true,
                supports_nvlink: true,
                nvlink_bandwidth_gbps: 896.0,
            });
        }

        // AMD Instinct MI300X & MI300A (CDNA 3, gfx942, 192GB / 128GB HBM3, 5.3 TB/s)
        if n.contains("mi300") {
            let is_apu = n.contains("mi300a");
            return Some(Self {
                name: name.to_string(),
                compute_capability: ComputeCapability::GFX_942_CDNA3,
                architecture: GpuArchitecture::Cdna3,
                form_factor: if is_apu {
                    HardwareFormFactor::ApuUnifiedMemoryWithNpu
                } else {
                    HardwareFormFactor::DatacenterOamInstinct
                },
                memory_tech: MemoryTechnology::Hbm3,
                tensor_core_gen: TensorCoreGeneration::AmdMfmaCdna3,
                sm_count: if is_apu { 228 } else { 304 },
                vram_capacity_bytes: if is_apu {
                    128 * 1024 * 1024 * 1024
                } else {
                    192 * 1024 * 1024 * 1024
                },
                memory_bus_width_bits: 8192,
                memory_bandwidth_gbps: 5300.0,
                l2_cache_bytes: 256 * 1024 * 1024,
                smem_per_sm_bytes: 128 * 1024,
                smem_per_block_bytes: 64 * 1024,
                max_threads_per_sm: 2048,
                supports_tma: false,
                supports_fp8: true,
                supports_nvfp4: false,
                supports_async_copy: true,
                supports_nvlink: true,
                nvlink_bandwidth_gbps: 896.0,
            });
        }

        // AMD Instinct MI250X & MI210 (CDNA 2, gfx90a)
        if n.contains("mi250") || n.contains("mi210") || n.contains("mi200") {
            return Some(Self {
                name: name.to_string(),
                compute_capability: ComputeCapability::GFX_90A_CDNA2,
                architecture: GpuArchitecture::Cdna2,
                form_factor: HardwareFormFactor::DatacenterOamInstinct,
                memory_tech: MemoryTechnology::Hbm2e,
                tensor_core_gen: TensorCoreGeneration::AmdMfmaCdna3,
                sm_count: 220,
                vram_capacity_bytes: 128 * 1024 * 1024 * 1024,
                memory_bus_width_bits: 8192,
                memory_bandwidth_gbps: 3200.0,
                l2_cache_bytes: 16 * 1024 * 1024,
                smem_per_sm_bytes: 64 * 1024,
                smem_per_block_bytes: 64 * 1024,
                max_threads_per_sm: 2048,
                supports_tma: false,
                supports_fp8: false,
                supports_nvfp4: false,
                supports_async_copy: true,
                supports_nvlink: true,
                nvlink_bandwidth_gbps: 800.0,
            });
        }

        // ==========================================
        // 2. AMD RADEON RX COMMERCIAL & WORKSTATION
        // ==========================================

        // AMD Radeon RX 7000 Series & PRO W7900 (RDNA 3, gfx1100)
        if n.contains("7900")
            || n.contains("7800")
            || n.contains("7700")
            || n.contains("7600")
            || n.contains("rx 7000")
            || n.contains("w7900")
            || n.contains("w7800")
        {
            let is_pro = n.contains("w7900") || n.contains("w7800") || n.contains("pro");
            let is_7900xtx = n.contains("7900 xtx") || n.contains("w7900");
            return Some(Self {
                name: name.to_string(),
                compute_capability: ComputeCapability::GFX_1100_RDNA3,
                architecture: GpuArchitecture::Rdna3,
                form_factor: if is_pro {
                    HardwareFormFactor::EnterpriseRackServer
                } else {
                    HardwareFormFactor::DesktopWorkstation
                },
                memory_tech: MemoryTechnology::Gddr6,
                tensor_core_gen: TensorCoreGeneration::AmdRdnaWmma,
                sm_count: if is_7900xtx {
                    96
                } else if n.contains("7800") {
                    60
                } else {
                    48
                },
                vram_capacity_bytes: if is_pro {
                    48 * 1024 * 1024 * 1024
                } else if is_7900xtx {
                    24 * 1024 * 1024 * 1024
                } else {
                    16 * 1024 * 1024 * 1024
                },
                memory_bus_width_bits: if is_7900xtx { 384 } else { 256 },
                memory_bandwidth_gbps: if is_7900xtx { 960.0 } else { 624.0 },
                l2_cache_bytes: 96 * 1024 * 1024,
                smem_per_sm_bytes: 64 * 1024,
                smem_per_block_bytes: 64 * 1024,
                max_threads_per_sm: 1024,
                supports_tma: false,
                supports_fp8: true, // WMMA supports FP8 on RDNA 3/3.5
                supports_nvfp4: false,
                supports_async_copy: true,
                supports_nvlink: false,
                nvlink_bandwidth_gbps: 0.0,
            });
        }

        // AMD Radeon RX 6000 Series & PRO W6800 (RDNA 2, gfx1030)
        if n.contains("6900")
            || n.contains("6800")
            || n.contains("6700")
            || n.contains("6600")
            || n.contains("rx 6000")
            || n.contains("w6800")
        {
            let is_6900 = n.contains("6900") || n.contains("6800");
            return Some(Self {
                name: name.to_string(),
                compute_capability: ComputeCapability::GFX_1030_RDNA2,
                architecture: GpuArchitecture::Rdna2,
                form_factor: HardwareFormFactor::DesktopWorkstation,
                memory_tech: MemoryTechnology::Gddr6,
                tensor_core_gen: TensorCoreGeneration::AmdRdnaWmma,
                sm_count: if is_6900 { 80 } else { 40 },
                vram_capacity_bytes: if is_6900 {
                    16 * 1024 * 1024 * 1024
                } else {
                    12 * 1024 * 1024 * 1024
                },
                memory_bus_width_bits: 256,
                memory_bandwidth_gbps: 512.0,
                l2_cache_bytes: 128 * 1024 * 1024, // Infinity Cache
                smem_per_sm_bytes: 64 * 1024,
                smem_per_block_bytes: 64 * 1024,
                max_threads_per_sm: 1024,
                supports_tma: false,
                supports_fp8: false,
                supports_nvfp4: false,
                supports_async_copy: false,
                supports_nvlink: false,
                nvlink_bandwidth_gbps: 0.0,
            });
        }

        // ==========================================
        // 3. AMD RYZEN AI APUS WITH EMBEDDED NPU
        // ==========================================

        // AMD Ryzen AI Max+ 395 & AI Max 300 Series (Strix Halo: 40 CUs RDNA 3.5 + XDNA 2 NPU 50+ TOPS, 256-bit LPDDR5X)
        if n.contains("395")
            || n.contains("ai max")
            || n.contains("strix halo")
            || n.contains("8060s")
            || n.contains("8050s")
        {
            let is_390 = n.contains("390") || n.contains("8050s");
            let is_385 = n.contains("385");
            let cus = if is_385 {
                24
            } else if is_390 {
                32
            } else {
                40 // Ryzen AI Max+ 395 / Radeon 8060S (40 CUs, 2560 shaders)
            };
            return Some(Self {
                name: name.to_string(),
                compute_capability: ComputeCapability::GFX_1151_RDNA3_5,
                architecture: GpuArchitecture::Rdna3_5,
                form_factor: HardwareFormFactor::ApuUnifiedMemoryWithNpu,
                memory_tech: MemoryTechnology::UnifiedLpddr5X,
                tensor_core_gen: TensorCoreGeneration::AmdXdnaNpuEngine,
                sm_count: cus,
                vram_capacity_bytes: 128 * 1024 * 1024 * 1024, // Up to 128GB unified LPDDR5X-8533 memory pool
                memory_bus_width_bits: 256,
                memory_bandwidth_gbps: 273.0,
                l2_cache_bytes: 64 * 1024 * 1024, // 32MB MALL Infinity Cache + CPU L3 Cache
                smem_per_sm_bytes: 64 * 1024,
                smem_per_block_bytes: 64 * 1024,
                max_threads_per_sm: 1024,
                supports_tma: false,
                supports_fp8: true,
                supports_nvfp4: true, // XDNA 2 supports Block FP4 / Microscaling
                supports_async_copy: true,
                supports_nvlink: false,
                nvlink_bandwidth_gbps: 0.0,
            });
        }

        // AMD Ryzen AI 300 Series (Strix Point: RDNA 3.5 + XDNA 2 NPU 50+ TOPS)
        if n.contains("ryzen ai")
            || n.contains("strix")
            || n.contains("hx 370")
            || n.contains("hx 375")
            || n.contains("kraken")
        {
            return Some(Self {
                name: name.to_string(),
                compute_capability: ComputeCapability::GFX_1150_RDNA3_5,
                architecture: GpuArchitecture::Rdna3_5,
                form_factor: HardwareFormFactor::ApuUnifiedMemoryWithNpu,
                memory_tech: MemoryTechnology::UnifiedLpddr5X,
                tensor_core_gen: TensorCoreGeneration::AmdXdnaNpuEngine,
                sm_count: 16,                                 // Radeon 890M RDNA 3.5 CUs
                vram_capacity_bytes: 64 * 1024 * 1024 * 1024, // System-shared unified memory pool
                memory_bus_width_bits: 256,
                memory_bandwidth_gbps: 256.0,
                l2_cache_bytes: 32 * 1024 * 1024,
                smem_per_sm_bytes: 64 * 1024,
                smem_per_block_bytes: 64 * 1024,
                max_threads_per_sm: 1024,
                supports_tma: false,
                supports_fp8: true,
                supports_nvfp4: true, // XDNA 2 supports Block FP4 / Microscaling
                supports_async_copy: true,
                supports_nvlink: false,
                nvlink_bandwidth_gbps: 0.0,
            });
        }

        // AMD Ryzen 8040 & 7040 Series (Hawk Point / Phoenix: RDNA 3 + XDNA 1 NPU)
        if n.contains("8040")
            || n.contains("7040")
            || n.contains("hawk point")
            || n.contains("phoenix")
            || n.contains("7840")
            || n.contains("8840")
        {
            return Some(Self {
                name: name.to_string(),
                compute_capability: ComputeCapability::GFX_1100_RDNA3,
                architecture: GpuArchitecture::Rdna3,
                form_factor: HardwareFormFactor::ApuUnifiedMemoryWithNpu,
                memory_tech: MemoryTechnology::UnifiedLpddr5X,
                tensor_core_gen: TensorCoreGeneration::AmdXdnaNpuEngine,
                sm_count: 12, // Radeon 780M CUs
                vram_capacity_bytes: 32 * 1024 * 1024 * 1024,
                memory_bus_width_bits: 128,
                memory_bandwidth_gbps: 120.0,
                l2_cache_bytes: 16 * 1024 * 1024,
                smem_per_sm_bytes: 64 * 1024,
                smem_per_block_bytes: 64 * 1024,
                max_threads_per_sm: 1024,
                supports_tma: false,
                supports_fp8: true,
                supports_nvfp4: false,
                supports_async_copy: true,
                supports_nvlink: false,
                nvlink_bandwidth_gbps: 0.0,
            });
        }

        // ==========================================
        // 4. NVIDIA HARDWARE MATRICES
        // ==========================================

        // Blackwell Datacenter & Superchips (B200, B300, GB200, GB300, B100)
        if n.contains("b200")
            || n.contains("gb200")
            || n.contains("b300")
            || n.contains("gb300")
            || n.contains("b100")
        {
            let is_superchip = n.contains("gb200") || n.contains("gb300");
            return Some(Self {
                name: name.to_string(),
                compute_capability: ComputeCapability::SM_100_BLACKWELL,
                architecture: GpuArchitecture::Blackwell,
                form_factor: if is_superchip {
                    HardwareFormFactor::SuperchipGraceBlackwell
                } else {
                    HardwareFormFactor::DatacenterSxmNvl
                },
                memory_tech: MemoryTechnology::Hbm3e,
                tensor_core_gen: TensorCoreGeneration::Gen5Blackwell,
                sm_count: 160,
                vram_capacity_bytes: 192 * 1024 * 1024 * 1024,
                memory_bus_width_bits: 8192,
                memory_bandwidth_gbps: 8000.0,
                l2_cache_bytes: 128 * 1024 * 1024,
                smem_per_sm_bytes: 256 * 1024,
                smem_per_block_bytes: 228 * 1024,
                max_threads_per_sm: 2048,
                supports_tma: true,
                supports_fp8: true,
                supports_nvfp4: true,
                supports_async_copy: true,
                supports_nvlink: true,
                nvlink_bandwidth_gbps: 1800.0,
            });
        }

        // Blackwell Commercial Series (RTX 5090, RTX 5080, RTX 5070, Mobile)
        if n.contains("5090") || n.contains("5080") || n.contains("5070") || n.contains("rtx 5000")
        {
            let is_mobile = n.contains("laptop") || n.contains("mobile");
            return Some(Self {
                name: name.to_string(),
                compute_capability: ComputeCapability::SM_120_BLACKWELL_CLIENT,
                architecture: GpuArchitecture::Blackwell,
                form_factor: if is_mobile {
                    HardwareFormFactor::LaptopMobile
                } else {
                    HardwareFormFactor::DesktopWorkstation
                },
                memory_tech: MemoryTechnology::Gddr7,
                tensor_core_gen: TensorCoreGeneration::Gen5Blackwell,
                sm_count: if n.contains("5090") { 170 } else { 84 },
                vram_capacity_bytes: if n.contains("5090") {
                    32 * 1024 * 1024 * 1024
                } else {
                    16 * 1024 * 1024 * 1024
                },
                memory_bus_width_bits: 512,
                memory_bandwidth_gbps: 1792.0,
                l2_cache_bytes: 96 * 1024 * 1024,
                smem_per_sm_bytes: 128 * 1024,
                smem_per_block_bytes: 99 * 1024,
                max_threads_per_sm: 1536,
                supports_tma: true,
                supports_fp8: true,
                supports_nvfp4: true,
                supports_async_copy: true,
                supports_nvlink: false,
                nvlink_bandwidth_gbps: 0.0,
            });
        }

        // Hopper Datacenter (H100, H200, H800, H20, DGX H100)
        if n.contains("h100") || n.contains("h200") || n.contains("h800") || n.contains("h20") {
            let is_h200 = n.contains("h200");
            return Some(Self {
                name: name.to_string(),
                compute_capability: ComputeCapability::SM_90_HOPPER,
                architecture: GpuArchitecture::Hopper,
                form_factor: HardwareFormFactor::DatacenterSxmNvl,
                memory_tech: if is_h200 {
                    MemoryTechnology::Hbm3e
                } else {
                    MemoryTechnology::Hbm3
                },
                tensor_core_gen: TensorCoreGeneration::Gen4HopperAda,
                sm_count: 132,
                vram_capacity_bytes: if is_h200 {
                    141 * 1024 * 1024 * 1024
                } else {
                    80 * 1024 * 1024 * 1024
                },
                memory_bus_width_bits: 5120,
                memory_bandwidth_gbps: if is_h200 { 4800.0 } else { 3350.0 },
                l2_cache_bytes: 50 * 1024 * 1024,
                smem_per_sm_bytes: 228 * 1024,
                smem_per_block_bytes: 228 * 1024,
                max_threads_per_sm: 2048,
                supports_tma: true,
                supports_fp8: true,
                supports_nvfp4: false,
                supports_async_copy: true,
                supports_nvlink: true,
                nvlink_bandwidth_gbps: 900.0,
            });
        }

        // Ada Lovelace Commercial & Enterprise (RTX 4090, 4080, 4070, 4060, RTX 6000 Ada, PRO 6000, L40S, L4)
        if n.contains("4090")
            || n.contains("4080")
            || n.contains("4070")
            || n.contains("4060")
            || n.contains("rtx 4000")
            || n.contains("6000 ada")
            || n.contains("l40")
            || n.contains("l4")
        {
            let is_enterprise = n.contains("6000") || n.contains("l40") || n.contains("l4");
            let is_mobile = n.contains("laptop") || n.contains("mobile");
            return Some(Self {
                name: name.to_string(),
                compute_capability: ComputeCapability::SM_89_ADA,
                architecture: GpuArchitecture::Ada,
                form_factor: if is_enterprise {
                    HardwareFormFactor::EnterpriseRackServer
                } else if is_mobile {
                    HardwareFormFactor::LaptopMobile
                } else {
                    HardwareFormFactor::DesktopWorkstation
                },
                memory_tech: MemoryTechnology::Gddr6X,
                tensor_core_gen: TensorCoreGeneration::Gen4HopperAda,
                sm_count: if n.contains("6000") || n.contains("l40s") {
                    142
                } else if n.contains("4090") {
                    128
                } else {
                    60
                },
                vram_capacity_bytes: if is_enterprise {
                    48 * 1024 * 1024 * 1024
                } else if n.contains("4090") {
                    24 * 1024 * 1024 * 1024
                } else {
                    16 * 1024 * 1024 * 1024
                },
                memory_bus_width_bits: 384,
                memory_bandwidth_gbps: 1008.0,
                l2_cache_bytes: 72 * 1024 * 1024,
                smem_per_sm_bytes: 100 * 1024,
                smem_per_block_bytes: 99 * 1024,
                max_threads_per_sm: 1536,
                supports_tma: false,
                supports_fp8: true,
                supports_nvfp4: false,
                supports_async_copy: true,
                supports_nvlink: false,
                nvlink_bandwidth_gbps: 0.0,
            });
        }

        // Ampere Datacenter & Enterprise (A100 SXM/PCIe 40GB/80GB, A30, A10, RTX A6000, RTX A5000, RTX A4000)
        if n.contains("a100") || n.contains("a30") || n.contains("a10") || n.contains("rtx a") {
            let is_a100 = n.contains("a100");
            let is_80gb = n.contains("80gb") || !n.contains("40gb");
            return Some(Self {
                name: name.to_string(),
                compute_capability: if is_a100 {
                    ComputeCapability::SM_80_AMPERE_DC
                } else {
                    ComputeCapability::SM_86_AMPERE_CLIENT
                },
                architecture: GpuArchitecture::Ampere,
                form_factor: if is_a100 {
                    HardwareFormFactor::DatacenterSxmNvl
                } else {
                    HardwareFormFactor::EnterpriseRackServer
                },
                memory_tech: if is_a100 {
                    MemoryTechnology::Hbm2e
                } else {
                    MemoryTechnology::Gddr6
                },
                tensor_core_gen: TensorCoreGeneration::Gen3Ampere,
                sm_count: if is_a100 { 108 } else { 84 },
                vram_capacity_bytes: if is_a100 && is_80gb {
                    80 * 1024 * 1024 * 1024
                } else if is_a100 {
                    40 * 1024 * 1024 * 1024
                } else {
                    48 * 1024 * 1024 * 1024
                },
                memory_bus_width_bits: if is_a100 { 5120 } else { 384 },
                memory_bandwidth_gbps: if is_a100 && is_80gb {
                    2039.0
                } else if is_a100 {
                    1555.0
                } else {
                    768.0
                },
                l2_cache_bytes: 40 * 1024 * 1024,
                smem_per_sm_bytes: 164 * 1024,
                smem_per_block_bytes: 163 * 1024,
                max_threads_per_sm: 2048,
                supports_tma: false,
                supports_fp8: false,
                supports_nvfp4: false,
                supports_async_copy: true,
                supports_nvlink: is_a100,
                nvlink_bandwidth_gbps: if is_a100 { 600.0 } else { 0.0 },
            });
        }

        // Ampere Commercial (RTX 3090, 3080, 3070, 3060, Mobile)
        if n.contains("3090")
            || n.contains("3080")
            || n.contains("3070")
            || n.contains("3060")
            || n.contains("rtx 3000")
        {
            let is_mobile = n.contains("laptop") || n.contains("mobile");
            return Some(Self {
                name: name.to_string(),
                compute_capability: ComputeCapability::SM_86_AMPERE_CLIENT,
                architecture: GpuArchitecture::Ampere,
                form_factor: if is_mobile {
                    HardwareFormFactor::LaptopMobile
                } else {
                    HardwareFormFactor::DesktopWorkstation
                },
                memory_tech: MemoryTechnology::Gddr6X,
                tensor_core_gen: TensorCoreGeneration::Gen3Ampere,
                sm_count: if n.contains("3090") {
                    82
                } else if n.contains("3080") {
                    68
                } else {
                    48
                },
                vram_capacity_bytes: if n.contains("3090") {
                    24 * 1024 * 1024 * 1024
                } else if n.contains("3080") {
                    10 * 1024 * 1024 * 1024
                } else {
                    8 * 1024 * 1024 * 1024
                },
                memory_bus_width_bits: 384,
                memory_bandwidth_gbps: 936.0,
                l2_cache_bytes: 6 * 1024 * 1024,
                smem_per_sm_bytes: 100 * 1024,
                smem_per_block_bytes: 99 * 1024,
                max_threads_per_sm: 1536,
                supports_tma: false,
                supports_fp8: false,
                supports_nvfp4: false,
                supports_async_copy: true,
                supports_nvlink: n.contains("3090"),
                nvlink_bandwidth_gbps: if n.contains("3090") { 112.5 } else { 0.0 },
            });
        }

        // Turing Commercial & Enterprise (RTX 2080, 2070, 2060, T4, Quadro RTX 4000/5000/6000/8000)
        if n.contains("2080")
            || n.contains("2070")
            || n.contains("2060")
            || n.contains("rtx 2000")
            || n.contains("t4")
            || n.contains("quadro rtx")
        {
            let is_mobile = n.contains("laptop") || n.contains("mobile");
            return Some(Self {
                name: name.to_string(),
                compute_capability: ComputeCapability::SM_75_TURING,
                architecture: GpuArchitecture::Turing,
                form_factor: if n.contains("t4") {
                    HardwareFormFactor::EnterpriseRackServer
                } else if is_mobile {
                    HardwareFormFactor::LaptopMobile
                } else {
                    HardwareFormFactor::DesktopWorkstation
                },
                memory_tech: MemoryTechnology::Gddr6,
                tensor_core_gen: TensorCoreGeneration::Gen2Turing,
                sm_count: if n.contains("2080") || n.contains("8000") {
                    72
                } else if n.contains("t4") {
                    40
                } else {
                    36
                },
                vram_capacity_bytes: if n.contains("8000") {
                    48 * 1024 * 1024 * 1024
                } else if n.contains("t4") {
                    16 * 1024 * 1024 * 1024
                } else {
                    8 * 1024 * 1024 * 1024
                },
                memory_bus_width_bits: 256,
                memory_bandwidth_gbps: 448.0,
                l2_cache_bytes: 4 * 1024 * 1024,
                smem_per_sm_bytes: 96 * 1024,
                smem_per_block_bytes: 64 * 1024,
                max_threads_per_sm: 1024,
                supports_tma: false,
                supports_fp8: false,
                supports_nvfp4: false,
                supports_async_copy: false,
                supports_nvlink: false,
                nvlink_bandwidth_gbps: 0.0,
            });
        }

        // Jetson Orin Edge Series & DGX Spark
        if n.contains("orin") || n.contains("jetson") || n.contains("spark") {
            let is_spark = n.contains("spark");
            return Some(Self {
                name: name.to_string(),
                compute_capability: ComputeCapability::SM_87_ORIN,
                architecture: GpuArchitecture::Orin,
                form_factor: if is_spark {
                    HardwareFormFactor::DgxStationSpark
                } else {
                    HardwareFormFactor::EdgeEmbedded
                },
                memory_tech: MemoryTechnology::UnifiedLpddr5X,
                tensor_core_gen: TensorCoreGeneration::Gen3Ampere,
                sm_count: if n.contains("64gb") || is_spark {
                    64
                } else {
                    32
                },
                vram_capacity_bytes: if is_spark {
                    128 * 1024 * 1024 * 1024
                } else {
                    64 * 1024 * 1024 * 1024
                },
                memory_bus_width_bits: 256,
                memory_bandwidth_gbps: 204.8,
                l2_cache_bytes: 4 * 1024 * 1024,
                smem_per_sm_bytes: 164 * 1024,
                smem_per_block_bytes: 163 * 1024,
                max_threads_per_sm: 1536,
                supports_tma: false,
                supports_fp8: false,
                supports_nvfp4: false,
                supports_async_copy: true,
                supports_nvlink: false,
                nvlink_bandwidth_gbps: 0.0,
            });
        }

        // ==========================================
        // 5. GOOGLE CLOUD & EDGE TPUS
        // ==========================================

        // Google TPU v6e "Trillium" (32GB HBM3, 920 TFLOPS BF16/FP8, 3rd Gen SparseCore, 3.2 Tbps ICI)
        if n.contains("trillium")
            || n.contains("tpu v6")
            || n.contains("tpuv6")
            || n.contains("v6e")
        {
            return Some(Self {
                name: name.to_string(),
                compute_capability: ComputeCapability::TPU_V6E_TRILLIUM,
                architecture: GpuArchitecture::GoogleTpuV6eTrillium,
                form_factor: HardwareFormFactor::DatacenterTpuPod2dTorus,
                memory_tech: MemoryTechnology::Hbm3,
                tensor_core_gen: TensorCoreGeneration::GoogleTpuMxuV6Trillium,
                sm_count: 8, // 4 Dual-MXUs + SparseCore
                vram_capacity_bytes: 32 * 1024 * 1024 * 1024,
                memory_bus_width_bits: 4096,
                memory_bandwidth_gbps: 1600.0,
                l2_cache_bytes: 64 * 1024 * 1024,
                smem_per_sm_bytes: 128 * 1024,
                smem_per_block_bytes: 128 * 1024,
                max_threads_per_sm: 2048,
                supports_tma: true,
                supports_fp8: true,
                supports_nvfp4: false,
                supports_async_copy: true,
                supports_nvlink: true, // 3.2 Tbps Inter-Chip Interconnect
                nvlink_bandwidth_gbps: 3200.0,
            });
        }

        // Google TPU v5p (95GB HBM2e, 459 TFLOPS BF16, 4800 Gbps 3D Torus ICI)
        if n.contains("tpu v5p") || n.contains("tpuv5p") || n.contains("v5p") {
            return Some(Self {
                name: name.to_string(),
                compute_capability: ComputeCapability::TPU_V5P,
                architecture: GpuArchitecture::GoogleTpuV5p,
                form_factor: HardwareFormFactor::DatacenterTpuPod3dTorus,
                memory_tech: MemoryTechnology::Hbm2e,
                tensor_core_gen: TensorCoreGeneration::GoogleTpuMxuV5,
                sm_count: 4, // 4 MXUs (128x128) + 2nd Gen SparseCore
                vram_capacity_bytes: 95 * 1024 * 1024 * 1024,
                memory_bus_width_bits: 4096,
                memory_bandwidth_gbps: 2760.0,
                l2_cache_bytes: 64 * 1024 * 1024,
                smem_per_sm_bytes: 128 * 1024,
                smem_per_block_bytes: 128 * 1024,
                max_threads_per_sm: 2048,
                supports_tma: true,
                supports_fp8: true,
                supports_nvfp4: false,
                supports_async_copy: true,
                supports_nvlink: true, // 4.8 Tbps 3D Torus ICI
                nvlink_bandwidth_gbps: 4800.0,
            });
        }

        // Google TPU v5e "ViperLite" (16GB HBM2, 197 TFLOPS INT8, 2D Torus ICI)
        if n.contains("tpu v5e")
            || n.contains("tpuv5e")
            || n.contains("v5e")
            || n.contains("viperlite")
            || n.contains("tpu v5")
        {
            return Some(Self {
                name: name.to_string(),
                compute_capability: ComputeCapability::TPU_V5E,
                architecture: GpuArchitecture::GoogleTpuV5e,
                form_factor: HardwareFormFactor::DatacenterTpuPod2dTorus,
                memory_tech: MemoryTechnology::Hbm2,
                tensor_core_gen: TensorCoreGeneration::GoogleTpuMxuV5,
                sm_count: 1, // 1 MXU (128x128)
                vram_capacity_bytes: 16 * 1024 * 1024 * 1024,
                memory_bus_width_bits: 1024,
                memory_bandwidth_gbps: 820.0,
                l2_cache_bytes: 32 * 1024 * 1024,
                smem_per_sm_bytes: 64 * 1024,
                smem_per_block_bytes: 64 * 1024,
                max_threads_per_sm: 1024,
                supports_tma: false,
                supports_fp8: true,
                supports_nvfp4: false,
                supports_async_copy: true,
                supports_nvlink: true, // 1.6 Tbps 2D Torus ICI
                nvlink_bandwidth_gbps: 1600.0,
            });
        }

        // Google TPU v4 / v4i (32GB HBM2, 275 TFLOPS BF16, 3D Torus OCS, SparseCore)
        if n.contains("tpu v4") || n.contains("tpuv4") || n.contains("v4i") || n.contains("tpu-v4")
        {
            return Some(Self {
                name: name.to_string(),
                compute_capability: ComputeCapability::TPU_V4,
                architecture: GpuArchitecture::GoogleTpuV4,
                form_factor: HardwareFormFactor::DatacenterTpuPod3dTorus,
                memory_tech: MemoryTechnology::Hbm2,
                tensor_core_gen: TensorCoreGeneration::GoogleTpuMxuV4,
                sm_count: 4, // 4 MXUs (128x128) + SparseCore
                vram_capacity_bytes: 32 * 1024 * 1024 * 1024,
                memory_bus_width_bits: 2048,
                memory_bandwidth_gbps: 1200.0,
                l2_cache_bytes: 32 * 1024 * 1024,
                smem_per_sm_bytes: 64 * 1024,
                smem_per_block_bytes: 64 * 1024,
                max_threads_per_sm: 1024,
                supports_tma: false,
                supports_fp8: false,
                supports_nvfp4: false,
                supports_async_copy: true,
                supports_nvlink: true, // 4.8 Tbps 3D Torus Optical Circuit Switched ICI
                nvlink_bandwidth_gbps: 4800.0,
            });
        }

        // Google TPU v3 (16GB/32GB HBM2, 123 TFLOPS BF16, Dual MXUs, Liquid Cooled)
        if n.contains("tpu v3") || n.contains("tpuv3") || n.contains("tpu-v3") {
            let is_32gb = n.contains("32gb");
            return Some(Self {
                name: name.to_string(),
                compute_capability: ComputeCapability::TPU_V3,
                architecture: GpuArchitecture::GoogleTpuV3,
                form_factor: HardwareFormFactor::DatacenterTpuPod2dTorus,
                memory_tech: MemoryTechnology::Hbm2,
                tensor_core_gen: TensorCoreGeneration::GoogleTpuMxuV3,
                sm_count: 2, // 2 MXUs (128x128)
                vram_capacity_bytes: if is_32gb {
                    32 * 1024 * 1024 * 1024
                } else {
                    16 * 1024 * 1024 * 1024
                },
                memory_bus_width_bits: 1024,
                memory_bandwidth_gbps: 900.0,
                l2_cache_bytes: 16 * 1024 * 1024,
                smem_per_sm_bytes: 32 * 1024,
                smem_per_block_bytes: 32 * 1024,
                max_threads_per_sm: 1024,
                supports_tma: false,
                supports_fp8: false,
                supports_nvfp4: false,
                supports_async_copy: true,
                supports_nvlink: true, // 650 Gbps 2D Torus ICI
                nvlink_bandwidth_gbps: 650.0,
            });
        }

        // Google TPU v2 (8GB/16GB HBM, 45 TFLOPS BF16, 128x128 MXU)
        if n.contains("tpu v2") || n.contains("tpuv2") || n.contains("tpu-v2") {
            let is_16gb = n.contains("16gb");
            return Some(Self {
                name: name.to_string(),
                compute_capability: ComputeCapability::TPU_V2,
                architecture: GpuArchitecture::GoogleTpuV2,
                form_factor: HardwareFormFactor::DatacenterTpuPod2dTorus,
                memory_tech: MemoryTechnology::Hbm,
                tensor_core_gen: TensorCoreGeneration::GoogleTpuMxuV2,
                sm_count: 1, // 1 MXU (128x128)
                vram_capacity_bytes: if is_16gb {
                    16 * 1024 * 1024 * 1024
                } else {
                    8 * 1024 * 1024 * 1024
                },
                memory_bus_width_bits: 1024,
                memory_bandwidth_gbps: 600.0,
                l2_cache_bytes: 16 * 1024 * 1024,
                smem_per_sm_bytes: 32 * 1024,
                smem_per_block_bytes: 32 * 1024,
                max_threads_per_sm: 1024,
                supports_tma: false,
                supports_fp8: false,
                supports_nvfp4: false,
                supports_async_copy: false,
                supports_nvlink: true, // 600 Gbps 2D Torus ICI
                nvlink_bandwidth_gbps: 600.0,
            });
        }

        // Google Coral Edge TPU (4 TOPS INT8, PCIe/USB/M.2)
        if n.contains("coral") || n.contains("edge tpu") || n.contains("edgetpu") {
            return Some(Self {
                name: name.to_string(),
                compute_capability: ComputeCapability::EDGE_TPU,
                architecture: GpuArchitecture::GoogleEdgeTpu,
                form_factor: HardwareFormFactor::EdgeTpuModule,
                memory_tech: MemoryTechnology::SramOnChip,
                tensor_core_gen: TensorCoreGeneration::GoogleEdgeTpuInt8Engine,
                sm_count: 1,                          // 64x64 INT8 Systolic Array
                vram_capacity_bytes: 8 * 1024 * 1024, // 8 MB on-chip SRAM / scratchpad
                memory_bus_width_bits: 64,
                memory_bandwidth_gbps: 32.0,
                l2_cache_bytes: 1024 * 1024,
                smem_per_sm_bytes: 64 * 1024,
                smem_per_block_bytes: 64 * 1024,
                max_threads_per_sm: 512,
                supports_tma: false,
                supports_fp8: false,
                supports_nvfp4: false,
                supports_async_copy: false,
                supports_nvlink: false,
                nvlink_bandwidth_gbps: 0.0,
            });
        }

        None
    }

    /// Autonomically chooses optimal GEMV / GEMM tile threadblock dimensions for this architecture.
    #[must_use]
    pub const fn optimal_gemv_threads(&self) -> u32 {
        match self.architecture {
            GpuArchitecture::GoogleTpuV6eTrillium
            | GpuArchitecture::GoogleTpuV5p
            | GpuArchitecture::GoogleTpuV4
            | GpuArchitecture::Cdna4
            | GpuArchitecture::Cdna3
            | GpuArchitecture::Blackwell
            | GpuArchitecture::Hopper => 256, // 8 warps / Wave64 x 4 / High-throughput 128x128 MXU
            GpuArchitecture::GoogleTpuV5e
            | GpuArchitecture::GoogleTpuV3
            | GpuArchitecture::GoogleTpuV2
            | GpuArchitecture::Rdna3
            | GpuArchitecture::Rdna3_5
            | GpuArchitecture::Rdna4
            | GpuArchitecture::Ada
            | GpuArchitecture::Ampere
            | GpuArchitecture::Orin
            | GpuArchitecture::Cdna2
            | GpuArchitecture::Cdna1
            | GpuArchitecture::XdnaNpu => 128, // 4 warps / Wave32 x 4 / Standard MXU
            GpuArchitecture::GoogleEdgeTpu
            | GpuArchitecture::Turing
            | GpuArchitecture::Volta
            | GpuArchitecture::Rdna1
            | GpuArchitecture::Rdna2 => 64, // 2 warps / 64x64 Edge Systolic
        }
    }
}
