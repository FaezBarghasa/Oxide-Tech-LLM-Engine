//! Intel AMX (Advanced Matrix Extensions) Tile Configuration and Compute Engine.
//!
//! Provides architectural abstractions for Intel Xeon / Sapphire Rapids AMX tile registers (`TMM0`..`TMM7`).

#![allow(
    clippy::similar_names,
    clippy::items_after_statements,
    clippy::undocumented_unsafe_blocks
)]

/// 64-byte aligned AMX tile configuration structure for `ldtilecfg`.
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy)]
pub struct TileConfig {
    pub palette_id: u8,
    pub start_row: u8,
    pub reserved: [u8; 14],
    pub colb: [u16; 16],
    pub rows: [u8; 16],
}

impl Default for TileConfig {
    fn default() -> Self {
        Self {
            palette_id: 1,
            start_row: 0,
            reserved: [0; 14],
            colb: [0; 16],
            rows: [0; 16],
        }
    }
}

/// Check if Intel AMX tile and INT8 matrix extensions are supported and enabled by host CPU.
#[must_use]
pub fn is_amx_supported() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        use core::arch::x86_64::__cpuid_count;
        // CPUID leaf 7, subleaf 0
        let res = __cpuid_count(7, 0);
        let has_tile = (res.edx & (1 << 24)) != 0;
        let has_int8 = (res.edx & (1 << 22)) != 0;
        has_tile && has_int8
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// Request Linux kernel permission for AMX tile data architecture state (`ARCH_REQ_XCOMP_PERM`).
#[must_use]
pub fn request_amx_permission() -> bool {
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    {
        unsafe extern "C" {
            fn syscall(number: i64, ...) -> i64;
        }
        const SYS_ARCH_PRCTL: i64 = 158;
        const ARCH_REQ_XCOMP_PERM: i32 = 0x1023;
        const XFEATURE_XTILEDATA: i64 = 18;

        // SAFETY: Invokes Linux arch_prctl syscall to initialize OS XTILE context.
        unsafe {
            let res = syscall(SYS_ARCH_PRCTL, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA);
            res == 0
        }
    }
    #[cfg(not(all(target_os = "linux", target_arch = "x86_64")))]
    {
        false
    }
}

/// Load AMX tile configuration into processor.
///
/// # Safety
/// AMX tile extensions must be supported and permission granted by the OS.
#[inline(always)]
pub unsafe fn load_tile_config(config: &TileConfig) {
    #[cfg(target_arch = "x86_64")]
    {
        use core::arch::asm;
        // SAFETY: Caller guarantees AMX availability.
        unsafe {
            asm!(
                "ldtilecfg [{cfg}]",
                cfg = in(reg) config,
                options(nostack)
            );
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = config;
    }
}

/// Release AMX tile registers, resetting hardware state to clean palette.
///
/// # Safety
/// AMX must be supported.
#[inline(always)]
pub unsafe fn release_tiles() {
    #[cfg(target_arch = "x86_64")]
    {
        use core::arch::asm;
        // SAFETY: Tilerelease resets tile registers to init state.
        unsafe {
            asm!("tilerelease", options(nostack));
        }
    }
}

/// Execute 16x64 x 64x16 AMX INT8 matrix multiplication: `C += A * B` (`tdpbusd`).
///
/// # Safety
/// Caller must guarantee pointers are valid, correctly configured in tile registers, and AMX active.
#[inline(always)]
pub unsafe fn tile_matmul_int8(
    a_ptr: *const u8,
    a_stride: usize,
    b_ptr: *const i8,
    b_stride: usize,
    c_ptr: *mut i32,
    c_stride: usize,
) {
    #[cfg(target_arch = "x86_64")]
    {
        use core::arch::asm;
        // SAFETY: Direct assembly execution of tileloadd, tdpbusd, and tilestored.
        unsafe {
            asm!(
                "tileloadd tmm0, [{c_ptr} + {c_stride}]",
                "tileloadd tmm1, [{a_ptr} + {a_stride}]",
                "tileloadd tmm2, [{b_ptr} + {b_stride}]",
                "tdpbusd tmm0, tmm1, tmm2",
                "tilestored [{c_ptr} + {c_stride}], tmm0",
                a_ptr = in(reg) a_ptr,
                a_stride = in(reg) a_stride,
                b_ptr = in(reg) b_ptr,
                b_stride = in(reg) b_stride,
                c_ptr = in(reg) c_ptr,
                c_stride = in(reg) c_stride,
                options(nostack)
            );
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = (a_ptr, a_stride, b_ptr, b_stride, c_ptr, c_stride);
    }
}
