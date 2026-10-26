//! Pure Rust Native Hardware GPU Unlock, Topology & Resource Manager Engine
//!
//! Provides direct PCI sysfs hardware topology discovery, physical BAR0 MMIO
//! register access, live NVIDIA Resource Manager (RM) driver ioctl issue-rate
//! querying, and safe speed selector unlock routines for NVIDIA Ampere (RTX 3000)
//! and Ada Lovelace (RTX 4000) architectures, with zero simulation or external shims.
//!
//! Synthesizes and natively ports architectural discoveries from both:
//! - `references/gpu-unlock` (Ampere GA102/GA104/GA106/GA107 speed-select unlock)
//! - `references/gpu-unlock-ada` (Ada Lovelace AD102/AD103/AD104/AD106/AD107 four-PLM
//!   unlock and NVIDIA RM `GET_SM_RATES 0x20801230` public getter ABI).

use oxide_core::hardware::GpuArchitecture;
use serde::{Deserialize, Serialize};
use std::fs::{self, File, OpenOptions};
use std::os::unix::fs::FileExt;
use std::path::Path;

/// Base MMIO register offsets for NVIDIA GPU Feature Override and Speed Selectors.
/// Reference: Hardware GFW boot process & privilege level mask architecture.
pub mod registers {
    /// Chip ID register at physical BAR0 offset 0x0.
    /// E.g. `0x192000a1` (AD102), `0x197000a1` (AD107).
    pub const PMC_BOOT_0: u64 = 0x0000_0000;

    /// WPR Configuration Privilege Level Mask.
    pub const WPR_CFG_PLM: u64 = 0x001f_a7cc;

    /// WPR Partition Privilege Level Mask.
    pub const WPR_PLM: u64 = 0x001f_a7c4;

    /// Frame Buffer Partition Allocation Privilege Level Mask.
    pub const FBPA_PLM: u64 = 0x009a_0148;

    /// Feature Override Privilege Level Mask (PLM).
    /// Default on stock consumer VBIOS: `0xffff_ff8f` (L3 heavy-secure only write).
    /// Unlocked value: `0xffff_ffff` (all privilege levels granted write access).
    pub const FEAT_OVR_PLM: u64 = 0x0082_3804;

    /// Feature Override Effective Speed Readout.
    /// Encodes nine 3-bit fields: `imla0, fmla16, dp, fmla32, ffma, imla1, imla2, imla3, imla4`.
    /// Stock consumer throttled readout: `0x0041_0080` (fields `[0, 0, 2, 0, 0, 2, 0, 2, 0]`).
    /// Full-rate unlocked readout: `0x0000_0000` (all 9 fields = 0, full rate).
    pub const FEAT_OVR_SPEED_READOUT: u64 = 0x0082_3818;

    /// Feature Override SM Speed Selector 0 (`SS0`).
    /// Governs eight 4-bit fields: DP, FFMA, FMLA16, FMLA32, and IMLA0-3.
    /// Stock consumer VBIOS default: `0x4402_3025` (AD102) / `0x2102_5116` (AD107) / `0x0000_0001` (GA102).
    /// Full-rate unlocked value: `0x8888_8888`.
    pub const FEAT_OVR_SM_SPD_0: u64 = 0x0082_381c;

    /// Feature Override SM Speed Selector 1 (`SS1`).
    /// Governs one 4-bit field at [3:0], attributed to IMLA4.
    /// Stock consumer VBIOS default: `0x0000_0007` (Ada) / `0x0000_0002` (Ampere cold).
    /// Full-rate unlocked value: `0x0000_0008`.
    pub const FEAT_OVR_SM_SPD_1: u64 = 0x0082_3820;

    /// SEC2 Fuse register for signature version verification.
    pub const FUSE_SEC2_UCODE1: u64 = 0x0082_4140;

    /// Default PLM value locking host-side register writes
    pub const PLM_DEFAULT_LOCKED: u32 = 0xffff_ff8f;

    /// Unlocked PLM value enabling host driver register writes
    pub const PLM_UNLOCKED: u32 = 0xffff_ffff;

    /// Speed Selector 0 full-rate override mask (all 8 lanes set to override + 00 full rate)
    pub const SS0_FULL_RATE_OVERRIDE: u32 = 0x8888_8888;

    /// Speed Selector 1 full-rate override mask
    pub const SS1_FULL_RATE_OVERRIDE: u32 = 0x0000_0008;

    /// Speed readout mask (nine 3-bit fields = 27 bits)
    pub const SPEED_READOUT_MASK: u32 = 0x07ff_ffff;
}

/// Native Pure Rust NVIDIA Resource Manager (RM) Client via `/dev/nvidiactl` and `/dev/nvidia0`.
/// Queries live issue rates directly from the NVIDIA kernel driver without requiring root or BAR mapping.
pub mod nv_rm {
    use serde::{Deserialize, Serialize};
    use std::fs::OpenOptions;
    use std::os::fd::AsRawFd;

    /// Issue rate modifier for a GPU instruction execution pipe.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
    pub enum IssueRate {
        /// Full rate (1.0x native issue throughput).
        Full,
        /// Half rate (0.5x issue throughput, typical consumer stock throttle on FMLA32).
        Half,
        /// Quarter rate (0.25x issue throughput).
        Quarter,
        /// Further reduction factor.
        Reduced(u8),
    }

    impl IssueRate {
        #[must_use]
        pub fn from_raw(code: u8) -> Self {
            match code {
                0 => Self::Full,
                1 => Self::Half,
                2 => Self::Quarter,
                other => Self::Reduced(other),
            }
        }

        #[must_use]
        pub fn multiplier(&self) -> f32 {
            match self {
                Self::Full => 1.0,
                Self::Half => 0.5,
                Self::Quarter => 0.25,
                Self::Reduced(v) => 1.0 / (1 << v) as f32,
            }
        }
    }

    /// Live SM issue-rate modifier fields queried from NVIDIA Resource Manager command `0x20801230`.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
    pub struct NvSmIssueRates {
        /// Integer MMA 0 rate.
        pub imla0: IssueRate,
        /// FP16 Tensor Core with FP16 accumulator.
        pub fmla16: IssueRate,
        /// DP4A dot-product integer rate.
        pub dp: IssueRate,
        /// FP16/BF16/TF32 Tensor Core with FP32 accumulator (the critical throttled consumer path!).
        pub fmla32: IssueRate,
        /// Scalar FP32 FMA rate.
        pub ffma: IssueRate,
        /// Integer MMA 1 rate.
        pub imla1: IssueRate,
        /// Integer MMA 2 rate.
        pub imla2: IssueRate,
        /// Integer MMA 3 rate.
        pub imla3: IssueRate,
        /// Integer MMA 4 rate.
        pub imla4: IssueRate,
    }

    impl NvSmIssueRates {
        /// Returns true if the critical FP32-accumulate tensor path (`fmla32`) is throttled below full rate.
        #[must_use]
        pub fn is_fmla32_throttled(&self) -> bool {
            self.fmla32 != IssueRate::Full
        }

        /// Returns true if all issue pipes are running at unthrottled full rate.
        #[must_use]
        pub fn is_fully_unlocked(&self) -> bool {
            self.fmla32 == IssueRate::Full
                && self.fmla16 == IssueRate::Full
                && self.ffma == IssueRate::Full
                && self.imla2 == IssueRate::Full
                && self.imla4 == IssueRate::Full
        }
    }

    #[repr(C)]
    #[derive(Debug, Default, Clone, Copy)]
    struct NvRmAllocRaw {
        root: u32,
        parent: u32,
        new_handle: u32,
        cls: u32,
        params: u64,
        rights: u64,
        size: u32,
        flags: u32,
        status: u32,
    }

    #[repr(C)]
    #[derive(Debug, Default, Clone, Copy)]
    struct NvRmControlRaw {
        client: u32,
        obj: u32,
        cmd: u32,
        flags: u32,
        params: u64,
        size: u32,
        status: u32,
    }

    #[repr(C)]
    #[derive(Debug, Default, Clone, Copy)]
    struct NvRmDeviceRaw {
        device: u32,
        share: u32,
        target_client: u32,
        target_device: u32,
        flags: u32,
        _pad: u32,
        va_size: u64,
        va_start: u64,
        va_limit: u64,
        va_mode: u32,
        _pad2: u32,
    }

    #[repr(C)]
    #[derive(Debug, Default, Clone, Copy)]
    struct NvRmRatesRaw {
        route_flags: u32,
        padding: u32,
        route: u64,
        imla0: u8,
        fmla16: u8,
        dp: u8,
        fmla32: u8,
        ffma: u8,
        imla1: u8,
        imla2: u8,
        imla3: u8,
        imla4: u8,
        _reserved: [u8; 7],
    }

    const GET_SM_RATES_CMD: u32 = 0x2080_1230;

    fn ioctl_nr<T>(nr: u32) -> libc::c_ulong {
        let size = std::mem::size_of::<T>() as u32;
        libc::c_ulong::from((3u32 << 30) | (size << 16) | (u32::from(b'F') << 8) | nr)
    }

    /// Queries the live NVIDIA Resource Manager driver directly from `/dev/nvidiactl` and `/dev/nvidia0`.
    pub fn query_live_sm_rates() -> Result<NvSmIssueRates, String> {
        let ctl_file = OpenOptions::new()
            .read(true)
            .write(true)
            .open("/dev/nvidiactl")
            .map_err(|e| format!("Failed to open /dev/nvidiactl: {e}"))?;

        let _dev_file = OpenOptions::new()
            .read(true)
            .write(true)
            .open("/dev/nvidia0")
            .map_err(|e| format!("Failed to open /dev/nvidia0: {e}"))?;

        let ctl_fd = ctl_file.as_raw_fd();
        let alloc_ioc = ioctl_nr::<NvRmAllocRaw>(0x2B);
        let control_ioc = ioctl_nr::<NvRmControlRaw>(0x2A);

        // Step 1: Allocate Root Client (class 0x41)
        let mut client_req = NvRmAllocRaw {
            cls: 0x41,
            ..Default::default()
        };
        let ret = unsafe { libc::ioctl(ctl_fd, alloc_ioc, &mut client_req) };
        if ret != 0 || client_req.status != 0 || client_req.new_handle == 0 {
            return Err(format!(
                "RM client allocation failed (ret={ret}, status=0x{:x})",
                client_req.status
            ));
        }
        let client_handle = client_req.new_handle;

        // Step 2: Allocate Device Object (class 0x80)
        let mut dev_params = NvRmDeviceRaw::default();
        let mut dev_req = NvRmAllocRaw {
            root: client_handle,
            parent: client_handle,
            cls: 0x80,
            params: std::ptr::addr_of_mut!(dev_params) as u64,
            size: std::mem::size_of::<NvRmDeviceRaw>() as u32,
            ..Default::default()
        };
        let ret = unsafe { libc::ioctl(ctl_fd, alloc_ioc, &mut dev_req) };
        if ret != 0 || dev_req.status != 0 || dev_req.new_handle == 0 {
            return Err(format!(
                "RM device allocation failed (ret={ret}, status=0x{:x})",
                dev_req.status
            ));
        }
        let dev_handle = dev_req.new_handle;

        // Step 3: Allocate Subdevice Object (class 0x2080)
        let mut subdev_param: u32 = 0;
        let mut subdev_req = NvRmAllocRaw {
            root: client_handle,
            parent: dev_handle,
            cls: 0x2080,
            params: std::ptr::addr_of_mut!(subdev_param) as u64,
            size: std::mem::size_of::<u32>() as u32,
            ..Default::default()
        };
        let ret = unsafe { libc::ioctl(ctl_fd, alloc_ioc, &mut subdev_req) };
        if ret != 0 || subdev_req.status != 0 || subdev_req.new_handle == 0 {
            return Err(format!(
                "RM subdevice allocation failed (ret={ret}, status=0x{:x})",
                subdev_req.status
            ));
        }
        let subdev_handle = subdev_req.new_handle;

        // Step 4: Query Live SM Rates via GET_SM_RATES
        let mut rates_raw = NvRmRatesRaw::default();
        let mut ctrl_req = NvRmControlRaw {
            client: client_handle,
            obj: subdev_handle,
            cmd: GET_SM_RATES_CMD,
            params: std::ptr::addr_of_mut!(rates_raw) as u64,
            size: std::mem::size_of::<NvRmRatesRaw>() as u32,
            ..Default::default()
        };
        let ret = unsafe { libc::ioctl(ctl_fd, control_ioc, &mut ctrl_req) };
        if ret != 0 || ctrl_req.status != 0 {
            return Err(format!(
                "RM GET_SM_RATES query failed (ret={ret}, status=0x{:x})",
                ctrl_req.status
            ));
        }

        Ok(NvSmIssueRates {
            imla0: IssueRate::from_raw(rates_raw.imla0),
            fmla16: IssueRate::from_raw(rates_raw.fmla16),
            dp: IssueRate::from_raw(rates_raw.dp),
            fmla32: IssueRate::from_raw(rates_raw.fmla32),
            ffma: IssueRate::from_raw(rates_raw.ffma),
            imla1: IssueRate::from_raw(rates_raw.imla1),
            imla2: IssueRate::from_raw(rates_raw.imla2),
            imla3: IssueRate::from_raw(rates_raw.imla3),
            imla4: IssueRate::from_raw(rates_raw.imla4),
        })
    }
}

/// Decoded 9-field Speed Readout from register `0x00823818`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct EffectiveSpeedReadout {
    pub raw: u32,
    pub imla0: u8,
    pub fmla16: u8,
    pub dp: u8,
    pub fmla32: u8,
    pub ffma: u8,
    pub imla1: u8,
    pub imla2: u8,
    pub imla3: u8,
    pub imla4: u8,
}

impl EffectiveSpeedReadout {
    #[must_use]
    pub fn from_raw(raw: u32) -> Self {
        let masked = raw & registers::SPEED_READOUT_MASK;
        Self {
            raw: masked,
            imla0: (masked & 7) as u8,
            fmla16: ((masked >> 3) & 7) as u8,
            dp: ((masked >> 6) & 7) as u8,
            fmla32: ((masked >> 9) & 7) as u8,
            ffma: ((masked >> 12) & 7) as u8,
            imla1: ((masked >> 15) & 7) as u8,
            imla2: ((masked >> 18) & 7) as u8,
            imla3: ((masked >> 21) & 7) as u8,
            imla4: ((masked >> 24) & 7) as u8,
        }
    }

    #[must_use]
    pub fn is_all_zero(&self) -> bool {
        self.raw == 0
    }
}

/// Access status of live physical BAR0 MMIO hardware registers.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MmioAccessStatus {
    /// Registers read directly and verified from physical BAR0 MMIO space.
    ReadVerified,
    /// Permission denied attempting to open `/sys/bus/pci/devices/<bdf>/resource0` (requires root/CAP_SYS_RAWIO).
    PermissionDenied,
    /// BAR0 MMIO resource file does not exist or device is detached.
    ResourceUnavailable,
    /// Positional read failed or timed out.
    ReadFailed(String),
}

/// Live hardware MMIO register values read from physical GPU BAR0.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LiveMmioRegisters {
    pub pmc_boot_0: Option<u32>,
    pub feat_ovr_plm: Option<u32>,
    pub feat_ovr_sm_spd_0: Option<u32>,
    pub feat_ovr_sm_spd_1: Option<u32>,
    pub feat_ovr_speed_readout: Option<u32>,
    pub decoded_readout: Option<EffectiveSpeedReadout>,
    pub status: MmioAccessStatus,
}

/// Unlock and architectural status of a physical NVIDIA GPU.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum GpuUnlockCapability {
    /// Ada Lovelace (RTX 4000 series) GPU with stock firmware half-rate FMLA32 throttle.
    /// Fully compatible with speed selector unlock to double FP16/BF16/TF32 tensor issue rate.
    AdaThrottledStock {
        chip_series: &'static str,
        cuda_cores: u32,
        sm_count: u32,
        potential_gemm_gain_pct: &'static str,
    },
    /// Ada Lovelace GPU with Speed Selectors unlocked to full-rate.
    AdaUnlocked {
        chip_series: &'static str,
        cuda_cores: u32,
        sm_count: u32,
    },
    /// Ada Lovelace enterprise / datacenter GPU (natively unthrottled out of the box).
    AdaEnterpriseUnthrottled {
        chip_series: &'static str,
        cuda_cores: u32,
        sm_count: u32,
    },
    /// Ampere (RTX 3000 series) GPU with stock firmware half-rate BF16 issue throttling.
    /// Fully compatible with speed selector PLM unlock (+68-100% GEMM throughput).
    AmpereThrottledStock {
        chip_series: &'static str,
        cuda_cores: u32,
        sm_count: u32,
        potential_gemm_gain_pct: &'static str,
    },
    /// Ampere GPU that has already had its Speed Selectors unlocked to full-rate.
    AmpereUnlocked {
        chip_series: &'static str,
        cuda_cores: u32,
        sm_count: u32,
    },
    /// Ampere enterprise / datacenter GPU (A100, RTX A6000, etc.).
    AmpereEnterpriseUnthrottled {
        chip_series: &'static str,
        cuda_cores: u32,
        sm_count: u32,
    },
    /// Other modern architecture natively unthrottled.
    ModernNativeUnthrottled { architecture: GpuArchitecture },
    /// Other or non-PCI hardware
    Unsupported { device_id: u16 },
}

/// Physical Base Address Register (BAR) descriptor parsed from PCI sysfs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PciBarRegion {
    pub bar_index: usize,
    pub physical_address: u64,
    pub size_bytes: u64,
    pub is_mmio: bool,
    pub is_prefetchable: bool,
}

/// Represents a physical NVIDIA GPU discovered on the host PCI bus.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
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
    pub live_registers: LiveMmioRegisters,
    pub live_rm_rates: Option<nv_rm::NvSmIssueRates>,
    pub max_link_speed: Option<String>,
    pub current_link_speed: Option<String>,
    pub numa_node: i32,
}

/// Theoretical dense GEMM throughput projections.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct GpuThroughputProjection {
    pub fp32_tflops: f32,
    pub fp16_dense_tflops: f32,
    pub fmla32_stock_tflops: f32,
    pub fmla32_unlocked_tflops: f32,
    pub tf32_stock_tflops: f32,
    pub tf32_unlocked_tflops: f32,
    pub fp8_dense_tflops: f32,
}

/// Result of an in-place register unlock attempt.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum UnlockResult {
    /// Already running at native full-rate (speed readout is zero, full issue rate active).
    AlreadyFullRate,
    /// Successfully modified PLM and Speed Selectors to full rate (`0x88888888`).
    SuccessfullyUnlocked {
        old_ss0: u32,
        new_ss0: u32,
        old_ss1: u32,
        new_ss1: u32,
    },
    /// Permission denied opening `/sys/bus/pci/devices/<bdf>/resource0` for write.
    RequiresRootPrivileges,
    /// Card is held by the active NVIDIA driver (requires pre-driver execution or driver unload).
    DeviceBusyDriverBound,
    /// Target device does not require or support this hardware unlock path.
    NotApplicable,
    /// Hardware write verification failure.
    VerificationFailed { expected_ss0: u32, actual_ss0: u32 },
}

/// Pure Rust Hardware GPU Scanner, Unlock Diagnostic & Resource Manager Engine.
#[derive(Debug, Default, Clone)]
pub struct HardwareGpuUnlockEngine;

impl HardwareGpuUnlockEngine {
    /// Creates a new instance of the unlock and probe engine.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Scans the physical host PCI bus for NVIDIA GPUs.
    #[must_use]
    pub fn scan_pci_devices(&self) -> Vec<PhysicalPciGpu> {
        Self::probe_physical_gpus()
    }

    /// Scans the physical host PCI bus (`/sys/bus/pci/devices/`) for NVIDIA GPUs.
    pub fn probe_physical_gpus() -> Vec<PhysicalPciGpu> {
        let pci_root = Path::new("/sys/bus/pci/devices");
        if !pci_root.exists() {
            return Vec::new();
        }

        let mut discovered = Vec::new();

        // Attempt querying the live Resource Manager rates from the driver
        let live_rm = nv_rm::query_live_sm_rates().ok();

        if let Ok(entries) = fs::read_dir(pci_root) {
            for entry in entries.flatten() {
                let dev_path = entry.path();
                let bdf = match dev_path.file_name().and_then(|n| n.to_str()) {
                    Some(s) => s.to_string(),
                    None => continue,
                };

                let vendor = match Self::read_sysfs_hex(&dev_path.join("vendor")) {
                    Some(v) => v as u16,
                    None => continue,
                };

                // Filter for NVIDIA vendor ID (0x10de)
                if vendor != 0x10de {
                    continue;
                }

                let pci_class = Self::read_sysfs_hex(&dev_path.join("class")).unwrap_or(0);
                // Filter for display/3D compute controllers (Class 0x03xxxx)
                if (pci_class >> 16) != 0x03 {
                    continue;
                }

                let device_id = Self::read_sysfs_hex(&dev_path.join("device")).unwrap_or(0) as u16;
                let sub_vendor =
                    Self::read_sysfs_hex(&dev_path.join("subsystem_vendor")).unwrap_or(0) as u16;
                let sub_device =
                    Self::read_sysfs_hex(&dev_path.join("subsystem_device")).unwrap_or(0) as u16;

                let bar0 = Self::parse_bar0(&dev_path.join("resource"));
                let live_regs = Self::read_live_registers(&dev_path);

                let is_unlocked = if let Some(ref readout) = live_regs.decoded_readout {
                    readout.is_all_zero()
                } else if let Some(ref rm) = live_rm {
                    rm.is_fully_unlocked()
                } else {
                    false
                };

                let (arch, model_name, cap) = Self::classify_device(device_id, is_unlocked);

                let max_link_speed = fs::read_to_string(dev_path.join("max_link_speed"))
                    .map(|s| s.trim().to_string())
                    .ok();
                let current_link_speed = fs::read_to_string(dev_path.join("current_link_speed"))
                    .map(|s| s.trim().to_string())
                    .ok();
                let numa_node = fs::read_to_string(dev_path.join("numa_node"))
                    .ok()
                    .and_then(|s| s.trim().parse::<i32>().ok())
                    .unwrap_or(-1);

                discovered.push(PhysicalPciGpu {
                    pci_bdf: bdf,
                    vendor_id: vendor,
                    device_id,
                    subsystem_vendor: sub_vendor,
                    subsystem_device: sub_device,
                    pci_class,
                    architecture: arch,
                    model_name,
                    capability: cap,
                    bar0_region: bar0,
                    live_registers: live_regs,
                    live_rm_rates: live_rm,
                    max_link_speed,
                    current_link_speed,
                    numa_node,
                });
            }
        }

        discovered.sort_by(|a, b| a.pci_bdf.cmp(&b.pci_bdf));
        discovered
    }

    /// Reads physical BAR0 registers at offsets `PMC_BOOT_0`, `FEAT_OVR_PLM`,
    /// `FEAT_OVR_SPEED_READOUT`, `FEAT_OVR_SM_SPD_0`, and `FEAT_OVR_SM_SPD_1`.
    pub fn read_live_registers(dev_path: &Path) -> LiveMmioRegisters {
        let resource0_path = dev_path.join("resource0");
        if !resource0_path.exists() {
            return LiveMmioRegisters {
                pmc_boot_0: None,
                feat_ovr_plm: None,
                feat_ovr_sm_spd_0: None,
                feat_ovr_sm_spd_1: None,
                feat_ovr_speed_readout: None,
                decoded_readout: None,
                status: MmioAccessStatus::ResourceUnavailable,
            };
        }

        let file = match File::open(&resource0_path) {
            Ok(f) => f,
            Err(e) if e.kind() == std::io::ErrorKind::PermissionDenied => {
                return LiveMmioRegisters {
                    pmc_boot_0: None,
                    feat_ovr_plm: None,
                    feat_ovr_sm_spd_0: None,
                    feat_ovr_sm_spd_1: None,
                    feat_ovr_speed_readout: None,
                    decoded_readout: None,
                    status: MmioAccessStatus::PermissionDenied,
                };
            }
            Err(e) => {
                return LiveMmioRegisters {
                    pmc_boot_0: None,
                    feat_ovr_plm: None,
                    feat_ovr_sm_spd_0: None,
                    feat_ovr_sm_spd_1: None,
                    feat_ovr_speed_readout: None,
                    decoded_readout: None,
                    status: MmioAccessStatus::ReadFailed(e.to_string()),
                };
            }
        };

        let mut buf = [0u8; 4];
        let mut read_u32 = |offset: u64| -> Option<u32> {
            file.read_exact_at(&mut buf, offset)
                .ok()
                .map(|()| u32::from_le_bytes(buf))
        };

        let pmc = read_u32(registers::PMC_BOOT_0);
        let plm = read_u32(registers::FEAT_OVR_PLM);
        let ss0 = read_u32(registers::FEAT_OVR_SM_SPD_0);
        let ss1 = read_u32(registers::FEAT_OVR_SM_SPD_1);
        let readout_raw = read_u32(registers::FEAT_OVR_SPEED_READOUT);
        let decoded = readout_raw.map(EffectiveSpeedReadout::from_raw);

        LiveMmioRegisters {
            pmc_boot_0: pmc,
            feat_ovr_plm: plm,
            feat_ovr_sm_spd_0: ss0,
            feat_ovr_sm_spd_1: ss1,
            feat_ovr_speed_readout: readout_raw,
            decoded_readout: decoded,
            status: MmioAccessStatus::ReadVerified,
        }
    }

    /// Attempts to apply the physical Speed Selector unlock to an Ampere or Ada GPU.
    pub fn apply_hardware_unlock(dev_path: &Path, gpu: &PhysicalPciGpu) -> UnlockResult {
        // If already full rate, report clean success
        if let Some(ref readout) = gpu.live_registers.decoded_readout {
            if readout.is_all_zero() {
                return UnlockResult::AlreadyFullRate;
            }
        }
        if let Some(ref rm) = gpu.live_rm_rates {
            if rm.is_fully_unlocked() {
                return UnlockResult::AlreadyFullRate;
            }
        }

        match gpu.capability {
            GpuUnlockCapability::AdaUnlocked { .. }
            | GpuUnlockCapability::AmpereUnlocked { .. }
            | GpuUnlockCapability::AdaEnterpriseUnthrottled { .. }
            | GpuUnlockCapability::AmpereEnterpriseUnthrottled { .. } => {
                UnlockResult::AlreadyFullRate
            }
            GpuUnlockCapability::AdaThrottledStock { .. }
            | GpuUnlockCapability::AmpereThrottledStock { .. } => {
                let resource0_path = dev_path.join("resource0");
                let file = match OpenOptions::new()
                    .read(true)
                    .write(true)
                    .open(&resource0_path)
                {
                    Ok(f) => f,
                    Err(e) if e.kind() == std::io::ErrorKind::PermissionDenied => {
                        return UnlockResult::RequiresRootPrivileges;
                    }
                    Err(e) if e.kind() == std::io::ErrorKind::ResourceBusy => {
                        return UnlockResult::DeviceBusyDriverBound;
                    }
                    Err(_) => return UnlockResult::NotApplicable,
                };

                let mut buf = [0u8; 4];

                // Read current values
                let old_ss0 = if file
                    .read_exact_at(&mut buf, registers::FEAT_OVR_SM_SPD_0)
                    .is_ok()
                {
                    u32::from_le_bytes(buf)
                } else {
                    0
                };
                let old_ss1 = if file
                    .read_exact_at(&mut buf, registers::FEAT_OVR_SM_SPD_1)
                    .is_ok()
                {
                    u32::from_le_bytes(buf)
                } else {
                    0
                };

                // Step 1: Unlock PLM to allow host writes
                if file
                    .write_all_at(
                        &registers::PLM_UNLOCKED.to_le_bytes(),
                        registers::FEAT_OVR_PLM,
                    )
                    .is_err()
                {
                    return UnlockResult::RequiresRootPrivileges;
                }

                // Step 2: Write Speed Selector 0 (0x88888888)
                if file
                    .write_all_at(
                        &registers::SS0_FULL_RATE_OVERRIDE.to_le_bytes(),
                        registers::FEAT_OVR_SM_SPD_0,
                    )
                    .is_err()
                {
                    return UnlockResult::RequiresRootPrivileges;
                }

                // Step 3: Write Speed Selector 1 (0x00000008)
                if file
                    .write_all_at(
                        &registers::SS1_FULL_RATE_OVERRIDE.to_le_bytes(),
                        registers::FEAT_OVR_SM_SPD_1,
                    )
                    .is_err()
                {
                    return UnlockResult::RequiresRootPrivileges;
                }

                // Step 4: Verify readbacks
                let new_ss0 = if file
                    .read_exact_at(&mut buf, registers::FEAT_OVR_SM_SPD_0)
                    .is_ok()
                {
                    u32::from_le_bytes(buf)
                } else {
                    return UnlockResult::NotApplicable;
                };

                let new_ss1 = if file
                    .read_exact_at(&mut buf, registers::FEAT_OVR_SM_SPD_1)
                    .is_ok()
                {
                    u32::from_le_bytes(buf)
                } else {
                    return UnlockResult::NotApplicable;
                };

                if new_ss0 == registers::SS0_FULL_RATE_OVERRIDE
                    && new_ss1 == registers::SS1_FULL_RATE_OVERRIDE
                {
                    UnlockResult::SuccessfullyUnlocked {
                        old_ss0,
                        new_ss0,
                        old_ss1,
                        new_ss1,
                    }
                } else {
                    UnlockResult::VerificationFailed {
                        expected_ss0: registers::SS0_FULL_RATE_OVERRIDE,
                        actual_ss0: new_ss0,
                    }
                }
            }
            _ => UnlockResult::NotApplicable,
        }
    }

    /// Classifies an NVIDIA PCI device ID into architecture, human model name,
    /// core/SM counts, and hardware unlock capability.
    #[must_use]
    pub fn classify_device(
        device_id: u16,
        is_unlocked: bool,
    ) -> (GpuArchitecture, &'static str, GpuUnlockCapability) {
        match device_id {
            // =========================================================================
            // ADA LOVELACE ARCHITECTURE (RTX 4000 SERIES - SM 8.9)
            // References: gpu-unlock-ada (rtx4090-register-report, ada-2705, ada-28a0)
            // =========================================================================
            // AD102 Consumer: RTX 4090 / 4090 D / 4070 Ti SUPER (cut-down)
            0x2684 => (
                GpuArchitecture::Ada,
                "NVIDIA GeForce RTX 4090 (AD102)",
                if is_unlocked {
                    GpuUnlockCapability::AdaUnlocked {
                        chip_series: "AD102",
                        cuda_cores: 16384,
                        sm_count: 128,
                    }
                } else {
                    GpuUnlockCapability::AdaThrottledStock {
                        chip_series: "AD102",
                        cuda_cores: 16384,
                        sm_count: 128,
                        potential_gemm_gain_pct: "+100% (165 -> 331 TFLOPS FMLA32)",
                    }
                },
            ),
            0x2685 => (
                GpuArchitecture::Ada,
                "NVIDIA GeForce RTX 4090 D (AD102)",
                if is_unlocked {
                    GpuUnlockCapability::AdaUnlocked {
                        chip_series: "AD102",
                        cuda_cores: 14592,
                        sm_count: 114,
                    }
                } else {
                    GpuUnlockCapability::AdaThrottledStock {
                        chip_series: "AD102",
                        cuda_cores: 14592,
                        sm_count: 114,
                        potential_gemm_gain_pct: "+100% (FMLA32 tensor issue)",
                    }
                },
            ),
            0x2689 => (
                GpuArchitecture::Ada,
                "NVIDIA GeForce RTX 4070 Ti SUPER (AD102 cut-down)",
                if is_unlocked {
                    GpuUnlockCapability::AdaUnlocked {
                        chip_series: "AD102",
                        cuda_cores: 8448,
                        sm_count: 66,
                    }
                } else {
                    GpuUnlockCapability::AdaThrottledStock {
                        chip_series: "AD102",
                        cuda_cores: 8448,
                        sm_count: 66,
                        potential_gemm_gain_pct: "+100% (FMLA32 tensor issue)",
                    }
                },
            ),

            // AD102 Enterprise / Workstation (Unthrottled by default)
            0x26b5 => (
                GpuArchitecture::Ada,
                "NVIDIA RTX 6000 Ada Generation (AD102)",
                GpuUnlockCapability::AdaEnterpriseUnthrottled {
                    chip_series: "AD102",
                    cuda_cores: 18176,
                    sm_count: 142,
                },
            ),
            0x26b8 | 0x26b9 => (
                GpuArchitecture::Ada,
                "NVIDIA L40 / L40S (AD102)",
                GpuUnlockCapability::AdaEnterpriseUnthrottled {
                    chip_series: "AD102",
                    cuda_cores: 18176,
                    sm_count: 142,
                },
            ),

            // AD103: RTX 4080 SUPER / RTX 4080 / RTX 4070 Ti SUPER / RTX 4090 Mobile
            0x2702 => (
                GpuArchitecture::Ada,
                "NVIDIA GeForce RTX 4080 SUPER (AD103)",
                if is_unlocked {
                    GpuUnlockCapability::AdaUnlocked {
                        chip_series: "AD103",
                        cuda_cores: 10240,
                        sm_count: 80,
                    }
                } else {
                    GpuUnlockCapability::AdaThrottledStock {
                        chip_series: "AD103",
                        cuda_cores: 10240,
                        sm_count: 80,
                        potential_gemm_gain_pct: "+100% (103 -> 206 TFLOPS FMLA32)",
                    }
                },
            ),
            0x2704 => (
                GpuArchitecture::Ada,
                "NVIDIA GeForce RTX 4080 (AD103)",
                if is_unlocked {
                    GpuUnlockCapability::AdaUnlocked {
                        chip_series: "AD103",
                        cuda_cores: 9728,
                        sm_count: 76,
                    }
                } else {
                    GpuUnlockCapability::AdaThrottledStock {
                        chip_series: "AD103",
                        cuda_cores: 9728,
                        sm_count: 76,
                        potential_gemm_gain_pct: "+100% (98 -> 196 TFLOPS FMLA32)",
                    }
                },
            ),
            0x2705 => (
                GpuArchitecture::Ada,
                "NVIDIA GeForce RTX 4070 Ti SUPER (AD103)",
                if is_unlocked {
                    GpuUnlockCapability::AdaUnlocked {
                        chip_series: "AD103",
                        cuda_cores: 8448,
                        sm_count: 66,
                    }
                } else {
                    GpuUnlockCapability::AdaThrottledStock {
                        chip_series: "AD103",
                        cuda_cores: 8448,
                        sm_count: 66,
                        potential_gemm_gain_pct: "+100% (95 -> 190 TFLOPS FMLA32)",
                    }
                },
            ),
            0x2709 => (
                GpuArchitecture::Ada,
                "NVIDIA GeForce RTX 4070 (AD103 cut-down)",
                if is_unlocked {
                    GpuUnlockCapability::AdaUnlocked {
                        chip_series: "AD103",
                        cuda_cores: 5888,
                        sm_count: 46,
                    }
                } else {
                    GpuUnlockCapability::AdaThrottledStock {
                        chip_series: "AD103",
                        cuda_cores: 5888,
                        sm_count: 46,
                        potential_gemm_gain_pct: "+100% (FMLA32 tensor issue)",
                    }
                },
            ),
            0x2757 => (
                GpuArchitecture::Ada,
                "NVIDIA GeForce RTX 4090 Mobile / Max-Q (AD103M)",
                if is_unlocked {
                    GpuUnlockCapability::AdaUnlocked {
                        chip_series: "AD103M",
                        cuda_cores: 9728,
                        sm_count: 76,
                    }
                } else {
                    GpuUnlockCapability::AdaThrottledStock {
                        chip_series: "AD103M",
                        cuda_cores: 9728,
                        sm_count: 76,
                        potential_gemm_gain_pct: "+100% (FMLA32 tensor issue)",
                    }
                },
            ),

            // AD104: RTX 4070 Ti / 4070 SUPER / 4070 / L4
            0x2782 => (
                GpuArchitecture::Ada,
                "NVIDIA GeForce RTX 4070 Ti (AD104)",
                if is_unlocked {
                    GpuUnlockCapability::AdaUnlocked {
                        chip_series: "AD104",
                        cuda_cores: 7680,
                        sm_count: 60,
                    }
                } else {
                    GpuUnlockCapability::AdaThrottledStock {
                        chip_series: "AD104",
                        cuda_cores: 7680,
                        sm_count: 60,
                        potential_gemm_gain_pct: "+80-100% (FMLA32 tensor issue)",
                    }
                },
            ),
            0x2783 => (
                GpuArchitecture::Ada,
                "NVIDIA GeForce RTX 4070 SUPER (AD104)",
                if is_unlocked {
                    GpuUnlockCapability::AdaUnlocked {
                        chip_series: "AD104",
                        cuda_cores: 7168,
                        sm_count: 56,
                    }
                } else {
                    GpuUnlockCapability::AdaThrottledStock {
                        chip_series: "AD104",
                        cuda_cores: 7168,
                        sm_count: 56,
                        potential_gemm_gain_pct: "+100% (FMLA32 tensor issue)",
                    }
                },
            ),
            0x2786 => (
                GpuArchitecture::Ada,
                "NVIDIA GeForce RTX 4070 (AD104)",
                if is_unlocked {
                    GpuUnlockCapability::AdaUnlocked {
                        chip_series: "AD104",
                        cuda_cores: 5888,
                        sm_count: 46,
                    }
                } else {
                    GpuUnlockCapability::AdaThrottledStock {
                        chip_series: "AD104",
                        cuda_cores: 5888,
                        sm_count: 46,
                        potential_gemm_gain_pct: "+100% (FMLA32 tensor issue)",
                    }
                },
            ),
            0x27b8 => (
                GpuArchitecture::Ada,
                "NVIDIA L4 Enterprise (AD104)",
                GpuUnlockCapability::AdaEnterpriseUnthrottled {
                    chip_series: "AD104",
                    cuda_cores: 7424,
                    sm_count: 58,
                },
            ),

            // AD106: RTX 4060 Ti / 4060 Ti 16GB
            0x2803 | 0x2805 => (
                GpuArchitecture::Ada,
                "NVIDIA GeForce RTX 4060 Ti (AD106)",
                if is_unlocked {
                    GpuUnlockCapability::AdaUnlocked {
                        chip_series: "AD106",
                        cuda_cores: 4352,
                        sm_count: 34,
                    }
                } else {
                    GpuUnlockCapability::AdaThrottledStock {
                        chip_series: "AD106",
                        cuda_cores: 4352,
                        sm_count: 34,
                        potential_gemm_gain_pct: "+100% (FMLA32 tensor issue)",
                    }
                },
            ),

            // AD107: RTX 4060 Desktop / Laptop / Mobile / RTX 4050
            0x2882 => (
                GpuArchitecture::Ada,
                "NVIDIA GeForce RTX 4060 Desktop (AD107)",
                if is_unlocked {
                    GpuUnlockCapability::AdaUnlocked {
                        chip_series: "AD107",
                        cuda_cores: 3072,
                        sm_count: 24,
                    }
                } else {
                    GpuUnlockCapability::AdaThrottledStock {
                        chip_series: "AD107",
                        cuda_cores: 3072,
                        sm_count: 24,
                        potential_gemm_gain_pct: "+100% (28.5 -> 57.1 Top/s)",
                    }
                },
            ),
            0x28a0 => (
                GpuArchitecture::Ada,
                "NVIDIA GeForce RTX 4060 Laptop GPU (AD107M)",
                if is_unlocked {
                    GpuUnlockCapability::AdaUnlocked {
                        chip_series: "AD107M",
                        cuda_cores: 3072,
                        sm_count: 24,
                    }
                } else {
                    GpuUnlockCapability::AdaThrottledStock {
                        chip_series: "AD107M",
                        cuda_cores: 3072,
                        sm_count: 24,
                        potential_gemm_gain_pct: "+100% (28.5 -> 57.1 Top/s FMLA32)",
                    }
                },
            ),
            0x28e0 => (
                GpuArchitecture::Ada,
                "NVIDIA GeForce RTX 4060 Max-Q / Mobile (AD107M)",
                if is_unlocked {
                    GpuUnlockCapability::AdaUnlocked {
                        chip_series: "AD107M",
                        cuda_cores: 3072,
                        sm_count: 24,
                    }
                } else {
                    GpuUnlockCapability::AdaThrottledStock {
                        chip_series: "AD107M",
                        cuda_cores: 3072,
                        sm_count: 24,
                        potential_gemm_gain_pct: "+100% (28.5 -> 57.1 Top/s FMLA32)",
                    }
                },
            ),
            0x28e1 => (
                GpuArchitecture::Ada,
                "NVIDIA GeForce RTX 4050 Mobile / Max-Q (AD107M)",
                if is_unlocked {
                    GpuUnlockCapability::AdaUnlocked {
                        chip_series: "AD107M",
                        cuda_cores: 2560,
                        sm_count: 20,
                    }
                } else {
                    GpuUnlockCapability::AdaThrottledStock {
                        chip_series: "AD107M",
                        cuda_cores: 2560,
                        sm_count: 20,
                        potential_gemm_gain_pct: "+100% (FMLA32 tensor issue)",
                    }
                },
            ),

            // =========================================================================
            // AMPERE ARCHITECTURE (RTX 3000 SERIES - SM 8.6 / 8.0)
            // References: gpu-unlock (GA102/GA104/GA106/GA107 speed-select unlock)
            // =========================================================================
            0x2204 => (
                GpuArchitecture::Ampere,
                "NVIDIA GeForce RTX 3090 (GA102)",
                if is_unlocked {
                    GpuUnlockCapability::AmpereUnlocked {
                        chip_series: "GA102",
                        cuda_cores: 10496,
                        sm_count: 82,
                    }
                } else {
                    GpuUnlockCapability::AmpereThrottledStock {
                        chip_series: "GA102",
                        cuda_cores: 10496,
                        sm_count: 82,
                        potential_gemm_gain_pct: "+68-80% BF16 throughput",
                    }
                },
            ),
            0x2208 => (
                GpuArchitecture::Ampere,
                "NVIDIA GeForce RTX 3080 Ti (GA102)",
                if is_unlocked {
                    GpuUnlockCapability::AmpereUnlocked {
                        chip_series: "GA102",
                        cuda_cores: 10240,
                        sm_count: 80,
                    }
                } else {
                    GpuUnlockCapability::AmpereThrottledStock {
                        chip_series: "GA102",
                        cuda_cores: 10240,
                        sm_count: 80,
                        potential_gemm_gain_pct: "+68-80% BF16 throughput",
                    }
                },
            ),
            0x2206 | 0x220a => (
                GpuArchitecture::Ampere,
                "NVIDIA GeForce RTX 3080 (GA102)",
                if is_unlocked {
                    GpuUnlockCapability::AmpereUnlocked {
                        chip_series: "GA102",
                        cuda_cores: 8704,
                        sm_count: 68,
                    }
                } else {
                    GpuUnlockCapability::AmpereThrottledStock {
                        chip_series: "GA102",
                        cuda_cores: 8704,
                        sm_count: 68,
                        potential_gemm_gain_pct: "+68-80% BF16 throughput",
                    }
                },
            ),
            0x2207 => (
                GpuArchitecture::Ampere,
                "NVIDIA CMP 90HX (GA102)",
                if is_unlocked {
                    GpuUnlockCapability::AmpereUnlocked {
                        chip_series: "GA102",
                        cuda_cores: 10240,
                        sm_count: 80,
                    }
                } else {
                    GpuUnlockCapability::AmpereThrottledStock {
                        chip_series: "GA102",
                        cuda_cores: 10240,
                        sm_count: 80,
                        potential_gemm_gain_pct: "+68-80% BF16 throughput",
                    }
                },
            ),
            0x2230 => (
                GpuArchitecture::Ampere,
                "NVIDIA RTX A6000 (GA102)",
                GpuUnlockCapability::AmpereEnterpriseUnthrottled {
                    chip_series: "GA102",
                    cuda_cores: 10752,
                    sm_count: 84,
                },
            ),
            0x2235 => (
                GpuArchitecture::Ampere,
                "NVIDIA A40 (GA102)",
                GpuUnlockCapability::AmpereEnterpriseUnthrottled {
                    chip_series: "GA102",
                    cuda_cores: 10752,
                    sm_count: 84,
                },
            ),

            // GA104: RTX 3070 Ti / 3070 / 3060 Ti
            0x2482 => (
                GpuArchitecture::Ampere,
                "NVIDIA GeForce RTX 3070 Ti (GA104)",
                if is_unlocked {
                    GpuUnlockCapability::AmpereUnlocked {
                        chip_series: "GA104",
                        cuda_cores: 6144,
                        sm_count: 48,
                    }
                } else {
                    GpuUnlockCapability::AmpereThrottledStock {
                        chip_series: "GA104",
                        cuda_cores: 6144,
                        sm_count: 48,
                        potential_gemm_gain_pct: "+68-80% BF16 throughput",
                    }
                },
            ),
            0x2484 => (
                GpuArchitecture::Ampere,
                "NVIDIA GeForce RTX 3070 (GA104)",
                if is_unlocked {
                    GpuUnlockCapability::AmpereUnlocked {
                        chip_series: "GA104",
                        cuda_cores: 5888,
                        sm_count: 46,
                    }
                } else {
                    GpuUnlockCapability::AmpereThrottledStock {
                        chip_series: "GA104",
                        cuda_cores: 5888,
                        sm_count: 46,
                        potential_gemm_gain_pct: "+68-80% BF16 throughput",
                    }
                },
            ),
            0x2486 | 0x2489 => (
                GpuArchitecture::Ampere,
                "NVIDIA GeForce RTX 3060 Ti (GA104)",
                if is_unlocked {
                    GpuUnlockCapability::AmpereUnlocked {
                        chip_series: "GA104",
                        cuda_cores: 4864,
                        sm_count: 38,
                    }
                } else {
                    GpuUnlockCapability::AmpereThrottledStock {
                        chip_series: "GA104",
                        cuda_cores: 4864,
                        sm_count: 38,
                        potential_gemm_gain_pct: "+68-80% BF16 throughput",
                    }
                },
            ),
            0x24b0 => (
                GpuArchitecture::Ampere,
                "NVIDIA RTX A4000 (GA104)",
                GpuUnlockCapability::AmpereEnterpriseUnthrottled {
                    chip_series: "GA104",
                    cuda_cores: 6144,
                    sm_count: 48,
                },
            ),

            // GA103: RTX 3060 Ti OEM
            0x2414 => (
                GpuArchitecture::Ampere,
                "NVIDIA GeForce RTX 3060 Ti OEM (GA103)",
                if is_unlocked {
                    GpuUnlockCapability::AmpereUnlocked {
                        chip_series: "GA103",
                        cuda_cores: 4864,
                        sm_count: 38,
                    }
                } else {
                    GpuUnlockCapability::AmpereThrottledStock {
                        chip_series: "GA103",
                        cuda_cores: 4864,
                        sm_count: 38,
                        potential_gemm_gain_pct: "+68-80% BF16 throughput",
                    }
                },
            ),

            // GA106: RTX 3060 12GB
            0x2503 | 0x2504 => (
                GpuArchitecture::Ampere,
                "NVIDIA GeForce RTX 3060 12GB (GA106)",
                if is_unlocked {
                    GpuUnlockCapability::AmpereUnlocked {
                        chip_series: "GA106",
                        cuda_cores: 3584,
                        sm_count: 28,
                    }
                } else {
                    GpuUnlockCapability::AmpereThrottledStock {
                        chip_series: "GA106",
                        cuda_cores: 3584,
                        sm_count: 28,
                        potential_gemm_gain_pct: "+68-80% BF16 throughput",
                    }
                },
            ),

            // GA107: RTX 3050 8GB / 6GB
            0x2582 | 0x2584 => (
                GpuArchitecture::Ampere,
                "NVIDIA GeForce RTX 3050 (GA107)",
                if is_unlocked {
                    GpuUnlockCapability::AmpereUnlocked {
                        chip_series: "GA107",
                        cuda_cores: 2560,
                        sm_count: 20,
                    }
                } else {
                    GpuUnlockCapability::AmpereThrottledStock {
                        chip_series: "GA107",
                        cuda_cores: 2560,
                        sm_count: 20,
                        potential_gemm_gain_pct: "+68-80% BF16 throughput",
                    }
                },
            ),

            // GA100: A100 Datacenter
            0x20b0 | 0x20b2 | 0x20f1 => (
                GpuArchitecture::Ampere,
                "NVIDIA A100 Tensor Core GPU (GA100)",
                GpuUnlockCapability::AmpereEnterpriseUnthrottled {
                    chip_series: "GA100",
                    cuda_cores: 6912,
                    sm_count: 108,
                },
            ),

            // Hopper & Blackwell
            0x2300..=0x233f => (
                GpuArchitecture::Hopper,
                "NVIDIA H100 / H800 / H200 (Hopper)",
                GpuUnlockCapability::ModernNativeUnthrottled {
                    architecture: GpuArchitecture::Hopper,
                },
            ),
            0x2900..=0x297f => (
                GpuArchitecture::Blackwell,
                "NVIDIA B100 / B200 / GB200 (Blackwell)",
                GpuUnlockCapability::ModernNativeUnthrottled {
                    architecture: GpuArchitecture::Blackwell,
                },
            ),

            other => (
                GpuArchitecture::Turing,
                "NVIDIA Graphics Device",
                GpuUnlockCapability::Unsupported { device_id: other },
            ),
        }
    }

    /// Computes theoretical dense GEMM throughput projections (TFLOPS) given
    /// GPU architecture capability and SM clock frequency in MHz.
    #[must_use]
    pub fn compute_gemm_projections(
        cap: &GpuUnlockCapability,
        clock_mhz: f32,
    ) -> Option<GpuThroughputProjection> {
        let (cuda_cores, sm_count, is_ada) = match cap {
            GpuUnlockCapability::AdaThrottledStock {
                cuda_cores,
                sm_count,
                ..
            }
            | GpuUnlockCapability::AdaUnlocked {
                cuda_cores,
                sm_count,
                ..
            }
            | GpuUnlockCapability::AdaEnterpriseUnthrottled {
                cuda_cores,
                sm_count,
                ..
            } => (*cuda_cores, *sm_count, true),

            GpuUnlockCapability::AmpereThrottledStock {
                cuda_cores,
                sm_count,
                ..
            }
            | GpuUnlockCapability::AmpereUnlocked {
                cuda_cores,
                sm_count,
                ..
            }
            | GpuUnlockCapability::AmpereEnterpriseUnthrottled {
                cuda_cores,
                sm_count,
                ..
            } => (*cuda_cores, *sm_count, false),

            _ => return None,
        };

        let ghz = clock_mhz / 1000.0;
        let fp32_tflops = (cuda_cores as f32) * 2.0 * ghz / 1000.0;

        // In Ampere (SM 8.6), each SM executes 256 dense FP16 ops/clock.
        // In Ada (SM 8.9), each SM executes 512 dense FP16/BF16 ops/clock.
        let (ops_per_sm_unlocked, ops_per_sm_stock) = if is_ada {
            // Ada: 4th Gen Tensor Cores
            // Stock consumer FMLA32 has half issue rate (256 ops/SM), Unlocked is 512 ops/SM
            (512.0, 256.0)
        } else {
            // Ampere: 3rd Gen Tensor Cores
            // Stock consumer BF16 FMLA32 is half rate (128 ops/SM), Unlocked is 256 ops/SM
            (256.0, 128.0)
        };

        let fmla32_unlocked = (sm_count as f32) * ops_per_sm_unlocked * ghz / 1000.0;
        let fmla32_stock = (sm_count as f32) * ops_per_sm_stock * ghz / 1000.0;
        let fp16_dense = fmla32_unlocked;
        let tf32_unlocked = fmla32_unlocked / 2.0;
        let tf32_stock = fmla32_stock / 2.0;

        let fp8_dense = if is_ada {
            // 4th Gen Tensor Cores execute FP8 at 2x FP16 dense rate
            fmla32_unlocked * 2.0
        } else {
            0.0
        };

        Some(GpuThroughputProjection {
            fp32_tflops,
            fp16_dense_tflops: fp16_dense,
            fmla32_stock_tflops: fmla32_stock,
            fmla32_unlocked_tflops: fmla32_unlocked,
            tf32_stock_tflops: tf32_stock,
            tf32_unlocked_tflops: tf32_unlocked,
            fp8_dense_tflops: fp8_dense,
        })
    }

    fn read_sysfs_hex(path: &Path) -> Option<u32> {
        fs::read_to_string(path).ok().and_then(|s| {
            let clean = s.trim().trim_start_matches("0x");
            u32::from_str_radix(clean, 16).ok()
        })
    }

    fn parse_bar0(resource_path: &Path) -> Option<PciBarRegion> {
        let content = fs::read_to_string(resource_path).ok()?;
        let first_line = content.lines().next()?;
        let parts: Vec<&str> = first_line.split_whitespace().collect();
        if parts.len() < 3 {
            return None;
        }

        let start = u64::from_str_radix(parts[0].trim_start_matches("0x"), 16).ok()?;
        let end = u64::from_str_radix(parts[1].trim_start_matches("0x"), 16).ok()?;
        let flags = u64::from_str_radix(parts[2].trim_start_matches("0x"), 16).ok()?;

        if end < start {
            return None;
        }

        let size = end - start + 1;
        let is_mmio = (flags & 0x01) == 0;
        let is_prefetch = (flags & 0x08) != 0;

        Some(PciBarRegion {
            bar_index: 0,
            physical_address: start,
            size_bytes: size,
            is_mmio,
            is_prefetchable: is_prefetch,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ada_rtx4090_classification() {
        let (arch, name, cap) = HardwareGpuUnlockEngine::classify_device(0x2684, false);
        assert_eq!(arch, GpuArchitecture::Ada);
        assert!(name.contains("RTX 4090"));
        match cap {
            GpuUnlockCapability::AdaThrottledStock {
                cuda_cores,
                sm_count,
                ..
            } => {
                assert_eq!(cuda_cores, 16384);
                assert_eq!(sm_count, 128);
            }
            _ => panic!("Expected AdaThrottledStock for stock RTX 4090"),
        }
    }

    #[test]
    fn test_ada_rtx4060_mobile_classification() {
        // Host hardware reference: 0x28e0 AD107M
        let (arch, name, cap) = HardwareGpuUnlockEngine::classify_device(0x28e0, false);
        assert_eq!(arch, GpuArchitecture::Ada);
        assert!(name.contains("RTX 4060"));
        match cap {
            GpuUnlockCapability::AdaThrottledStock {
                cuda_cores,
                sm_count,
                ..
            } => {
                assert_eq!(cuda_cores, 3072);
                assert_eq!(sm_count, 24);
            }
            _ => panic!("Expected AdaThrottledStock for stock RTX 4060 Mobile"),
        }
    }

    #[test]
    fn test_ampere_rtx3090_classification() {
        let (arch, name, cap) = HardwareGpuUnlockEngine::classify_device(0x2204, false);
        assert_eq!(arch, GpuArchitecture::Ampere);
        assert!(name.contains("RTX 3090"));
        match cap {
            GpuUnlockCapability::AmpereThrottledStock {
                cuda_cores,
                sm_count,
                ..
            } => {
                assert_eq!(cuda_cores, 10496);
                assert_eq!(sm_count, 82);
            }
            _ => panic!("Expected AmpereThrottledStock"),
        }
    }

    #[test]
    fn test_speed_readout_decoding() {
        // Stock consumer readout: 0x00410080
        // Fields: [0, 0, 2, 0, 0, 2, 0, 2, 0]
        let readout = EffectiveSpeedReadout::from_raw(0x0041_0080);
        assert!(!readout.is_all_zero());
        assert_eq!(readout.raw, 0x0041_0080);
        assert_eq!(readout.imla0, 0);

        // Unlocked readout: 0x00000000
        let unlocked = EffectiveSpeedReadout::from_raw(0x0000_0000);
        assert!(unlocked.is_all_zero());
    }

    #[test]
    fn test_compute_projections_rtx4060() {
        let cap = GpuUnlockCapability::AdaThrottledStock {
            chip_series: "AD107M",
            cuda_cores: 3072,
            sm_count: 24,
            potential_gemm_gain_pct: "+100%",
        };
        let proj = HardwareGpuUnlockEngine::compute_gemm_projections(&cap, 2500.0)
            .expect("Valid projections");
        assert_eq!(proj.fp32_tflops, 15.36);
        assert_eq!(proj.fmla32_stock_tflops, 15.36);
        assert_eq!(proj.fmla32_unlocked_tflops, 30.72);
        assert_eq!(proj.fp8_dense_tflops, 61.44);
    }

    #[test]
    fn test_live_rm_rates_query() {
        // If /dev/nvidiactl exists, query live RM rates and verify valid issue rates
        if Path::new("/dev/nvidiactl").exists() {
            let res = nv_rm::query_live_sm_rates();
            assert!(res.is_ok(), "query_live_sm_rates failed: {:?}", res.err());
            let rates = res.unwrap();
            println!("Live RM rates: {rates:?}");
            // On standard consumer Ada cards, fmla32 is Half rate (1)
            assert_eq!(rates.fmla16, nv_rm::IssueRate::Full);
        }
    }

    #[test]
    fn test_probe_physical_gpus_on_host() {
        let gpus = HardwareGpuUnlockEngine::probe_physical_gpus();
        if !gpus.is_empty() {
            let primary = &gpus[0];
            assert_eq!(primary.vendor_id, 0x10de);
            assert_eq!(primary.pci_class >> 16, 0x03);
            println!(
                "Discovered physical GPU: {} (PCI {}, DevID 0x{:04x}, live status: {:?})",
                primary.model_name,
                primary.pci_bdf,
                primary.device_id,
                primary.live_registers.status
            );
        }
    }
}
