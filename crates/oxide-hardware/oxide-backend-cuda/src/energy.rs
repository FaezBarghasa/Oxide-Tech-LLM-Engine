//! High-precision GPU Energy Measurement & Cold-Start Joules Profiler.
//!
//! Provides dynamic runtime NVML symbol resolution (`libnvidia-ml.so.1`), monotonic
//! hardware energy counter extraction (`nvmlDeviceGetTotalEnergyConsumption`), and
//! background numerical integration of instantaneous milliwatts (`nvmlDeviceGetPowerUsage`).
//! Measures cold-start Joules, average power wattage, and tokens-per-Joule efficiency metrics.

use std::ffi::CStr;
use std::os::raw::{c_char, c_int, c_uint, c_void};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

/// NVML return codes.
pub const NVML_SUCCESS: c_int = 0;
pub const NVML_ERROR_UNINITIALIZED: c_int = 1;
pub const NVML_ERROR_INVALID_ARGUMENT: c_int = 2;
pub const NVML_ERROR_NOT_SUPPORTED: c_int = 3;
pub const NVML_ERROR_NO_PERMISSION: c_int = 4;
pub const NVML_ERROR_ALREADY_INITIALIZED: c_int = 5;
pub const NVML_ERROR_NOT_FOUND: c_int = 6;
pub const NVML_ERROR_INSUFFICIENT_SIZE: c_int = 7;
pub const NVML_ERROR_INSUFFICIENT_POWER: c_int = 8;
pub const NVML_ERROR_DRIVER_NOT_LOADED: c_int = 9;
pub const NVML_ERROR_TIMEOUT: c_int = 10;
pub const NVML_ERROR_UNKNOWN: c_int = 999;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NvmlDevice(pub *mut c_void);

type FnNvmlInit = unsafe extern "C" fn() -> c_int;
type FnNvmlShutdown = unsafe extern "C" fn() -> c_int;
type FnNvmlDeviceGetHandleByIndex = unsafe extern "C" fn(c_uint, *mut *mut c_void) -> c_int;
type FnNvmlDeviceGetTotalEnergyConsumption = unsafe extern "C" fn(*mut c_void, *mut u64) -> c_int;
type FnNvmlDeviceGetPowerUsage = unsafe extern "C" fn(*mut c_void, *mut c_uint) -> c_int;
type FnNvmlErrorString = unsafe extern "C" fn(c_int) -> *const c_char;

/// Dynamic NVML runtime function pointer table.
#[derive(Debug)]
pub struct DynamicNvml {
    lib_handle: *mut c_void,
    fn_shutdown: FnNvmlShutdown,
    fn_get_handle: FnNvmlDeviceGetHandleByIndex,
    fn_get_energy: Option<FnNvmlDeviceGetTotalEnergyConsumption>,
    fn_get_power: Option<FnNvmlDeviceGetPowerUsage>,
    fn_error_string: FnNvmlErrorString,
    initialized: bool,
}

// SAFETY: Function pointers dynamically resolved from libnvidia-ml.so are process-wide.
unsafe impl Send for DynamicNvml {}
unsafe impl Sync for DynamicNvml {}

impl DynamicNvml {
    /// Attempts to dynamically load `libnvidia-ml.so.1` or `libnvidia-ml.so`.
    pub fn try_load() -> Option<Self> {
        let candidate_libs: &[&[u8]] = &[b"libnvidia-ml.so.1\0", b"libnvidia-ml.so\0"];

        let mut handle = std::ptr::null_mut();
        for &lib in candidate_libs {
            // SAFETY: Loading dynamic library safely with RTLD_NOW | RTLD_LOCAL.
            let ptr =
                unsafe { libc::dlopen(lib.as_ptr().cast(), libc::RTLD_NOW | libc::RTLD_LOCAL) };
            if !ptr.is_null() {
                handle = ptr;
                break;
            }
        }

        if handle.is_null() {
            return None;
        }

        unsafe {
            let fn_init: FnNvmlInit = Self::dlsym_opt(handle, b"nvmlInit_v2\0")
                .or_else(|| Self::dlsym_opt(handle, b"nvmlInit\0"))?;

            let fn_shutdown: FnNvmlShutdown = Self::dlsym_opt(handle, b"nvmlShutdown\0")?;

            let fn_get_handle: FnNvmlDeviceGetHandleByIndex =
                Self::dlsym_opt(handle, b"nvmlDeviceGetHandleByIndex_v2\0")
                    .or_else(|| Self::dlsym_opt(handle, b"nvmlDeviceGetHandleByIndex\0"))?;

            let fn_get_energy: Option<FnNvmlDeviceGetTotalEnergyConsumption> =
                Self::dlsym_opt(handle, b"nvmlDeviceGetTotalEnergyConsumption\0");

            let fn_get_power: Option<FnNvmlDeviceGetPowerUsage> =
                Self::dlsym_opt(handle, b"nvmlDeviceGetPowerUsage\0");

            let fn_error_string: FnNvmlErrorString = Self::dlsym_opt(handle, b"nvmlErrorString\0")?;

            let ret = (fn_init)();
            let initialized = ret == NVML_SUCCESS || ret == NVML_ERROR_ALREADY_INITIALIZED;

            if !initialized {
                libc::dlclose(handle);
                return None;
            }

            Some(Self {
                lib_handle: handle,
                fn_shutdown,
                fn_get_handle,
                fn_get_energy,
                fn_get_power,
                fn_error_string,
                initialized,
            })
        }
    }

    unsafe fn dlsym_opt<T: Copy>(handle: *mut c_void, symbol: &[u8]) -> Option<T> {
        // SAFETY: dlerror and dlsym are libc dynamic symbol lookup APIs.
        unsafe {
            let sym = libc::dlsym(handle, symbol.as_ptr().cast());
            if sym.is_null() {
                None
            } else {
                Some(std::mem::transmute_copy(&sym))
            }
        }
    }

    /// Resolves the device handle for `device_index`.
    pub fn get_device_handle(&self, device_index: u32) -> Option<NvmlDevice> {
        if !self.initialized {
            return None;
        }
        let mut dev: *mut c_void = std::ptr::null_mut();
        // SAFETY: Calling dynamically resolved nvmlDeviceGetHandleByIndex with valid pointer.
        let ret = unsafe { (self.fn_get_handle)(device_index as c_uint, &raw mut dev) };
        if ret == NVML_SUCCESS && !dev.is_null() {
            Some(NvmlDevice(dev))
        } else {
            None
        }
    }

    /// Reads monotonic millijoules if supported.
    pub fn get_total_energy_mj(&self, device: NvmlDevice) -> Option<u64> {
        let fn_get_energy = self.fn_get_energy?;
        let mut energy_mj: u64 = 0;
        // SAFETY: Calling nvmlDeviceGetTotalEnergyConsumption with valid device handle.
        let ret = unsafe { (fn_get_energy)(device.0, &raw mut energy_mj) };
        if ret == NVML_SUCCESS {
            Some(energy_mj)
        } else {
            None
        }
    }

    /// Reads instantaneous power consumption in milliwatts.
    pub fn get_power_usage_mw(&self, device: NvmlDevice) -> Option<u32> {
        let fn_get_power = self.fn_get_power?;
        let mut power_mw: c_uint = 0;
        // SAFETY: Calling nvmlDeviceGetPowerUsage with valid device handle.
        let ret = unsafe { (fn_get_power)(device.0, &raw mut power_mw) };
        if ret == NVML_SUCCESS {
            Some(power_mw as u32)
        } else {
            None
        }
    }

    pub fn error_string(&self, code: c_int) -> &'static str {
        // SAFETY: nvmlErrorString returns a valid static nul-terminated string pointer.
        unsafe {
            let ptr = (self.fn_error_string)(code);
            if ptr.is_null() {
                "Unknown NVML error"
            } else {
                CStr::from_ptr(ptr)
                    .to_str()
                    .unwrap_or("Non-utf8 NVML error")
            }
        }
    }
}

impl Drop for DynamicNvml {
    fn drop(&mut self) {
        if self.initialized {
            // SAFETY: Shutting down NVML cleanly and closing library handle.
            unsafe {
                (self.fn_shutdown)();
                if !self.lib_handle.is_null() {
                    libc::dlclose(self.lib_handle);
                }
            }
        }
    }
}

/// Methodology used to acquire energy measurement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EnergyMethod {
    /// Monotonic hardware counter (`nvmlDeviceGetTotalEnergyConsumption`, Volta+).
    TotalEnergyCounter,
    /// Numerical integration of polled instantaneous power (`nvmlDeviceGetPowerUsage`).
    PolledPower,
    /// Fallback or unavailable on current device/environment.
    Unavailable,
}

impl EnergyMethod {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::TotalEnergyCounter => "total_energy_counter",
            Self::PolledPower => "polled_power",
            Self::Unavailable => "unavailable",
        }
    }
}

/// Snapshot of energy and power metrics over an elapsed duration.
#[derive(Debug, Clone, PartialEq)]
pub struct EnergyMeasurement {
    /// Cumulative energy consumed during the measured window in Joules (J).
    pub joules: f64,
    /// Average power consumption during the window in Watts (W).
    pub average_watts: f64,
    /// Elapsed wall-clock duration of the window.
    pub elapsed: Duration,
    /// Energy measurement methodology used.
    pub method: EnergyMethod,
}

impl EnergyMeasurement {
    /// Computes token generation efficiency in tokens per Joule (tok/J).
    #[must_use]
    pub fn tokens_per_joule(&self, tokens: usize) -> f64 {
        if self.joules > 0.0 {
            tokens as f64 / self.joules
        } else {
            0.0
        }
    }
}

enum SamplerBackend {
    Counter {
        nvml: Arc<DynamicNvml>,
        device: NvmlDevice,
        start_mj: u64,
    },
    Polled {
        stop_signal: Arc<AtomicBool>,
        joules_accum: Arc<Mutex<f64>>,
        _join_handle: Option<JoinHandle<()>>,
    },
    Unavailable,
}

/// Active GPU energy sampler.
pub struct EnergySampler {
    start_time: Instant,
    backend: SamplerBackend,
}

impl std::fmt::Debug for EnergySampler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EnergySampler")
            .field("start_time", &self.start_time)
            .finish()
    }
}

impl EnergySampler {
    /// Starts energy sampling for the specified GPU device index (e.g. 0).
    ///
    /// Automatically selects hardware `TotalEnergyCounter` if supported by
    /// the GPU architecture, falling back to background polled power integration
    /// if only instantaneous milliwatts are supported, or `Unavailable` if NVML
    /// cannot be loaded.
    #[must_use]
    pub fn start(device_index: u32) -> Self {
        let start_time = Instant::now();

        let nvml = match DynamicNvml::try_load() {
            Some(loaded) => Arc::new(loaded),
            None => {
                return Self {
                    start_time,
                    backend: SamplerBackend::Unavailable,
                };
            }
        };

        let Some(device) = nvml.get_device_handle(device_index) else {
            return Self {
                start_time,
                backend: SamplerBackend::Unavailable,
            };
        };

        // Try monotonic hardware energy counter first (Volta+, Ada Lovelace RTX 4060, Hopper, etc.)
        if let Some(start_mj) = nvml.get_total_energy_mj(device) {
            return Self {
                start_time,
                backend: SamplerBackend::Counter {
                    nvml,
                    device,
                    start_mj,
                },
            };
        }

        // Fallback: spawn background thread polling instantaneous milliwatts at 10ms cadence
        if nvml.get_power_usage_mw(device).is_some() {
            let stop_signal = Arc::new(AtomicBool::new(false));
            let joules_accum = Arc::new(Mutex::new(0.0));

            let thread_stop = Arc::clone(&stop_signal);
            let thread_accum = Arc::clone(&joules_accum);
            let thread_nvml = Arc::clone(&nvml);

            let join_handle = thread::Builder::new()
                .name("oxide-energy-poller".to_string())
                .spawn(move || {
                    let Some(thread_device) = thread_nvml.get_device_handle(device_index) else {
                        return;
                    };
                    let mut last_tick = Instant::now();
                    while !thread_stop.load(Ordering::Relaxed) {
                        thread::sleep(Duration::from_millis(10));
                        let now = Instant::now();
                        let dt = now.duration_since(last_tick).as_secs_f64();
                        last_tick = now;

                        if let Some(mw) = thread_nvml.get_power_usage_mw(thread_device) {
                            let joules_step = (f64::from(mw) / 1000.0) * dt;
                            if let Ok(mut lock) = thread_accum.lock() {
                                *lock += joules_step;
                            }
                        }
                    }
                })
                .ok();

            return Self {
                start_time,
                backend: SamplerBackend::Polled {
                    stop_signal,
                    joules_accum,
                    _join_handle: join_handle,
                },
            };
        }

        Self {
            start_time,
            backend: SamplerBackend::Unavailable,
        }
    }

    /// Measures the cumulative energy and average power since `start()`.
    #[must_use]
    pub fn measure(&self) -> Option<EnergyMeasurement> {
        let elapsed = self.start_time.elapsed();
        let elapsed_secs = elapsed.as_secs_f64().max(1e-6);

        match &self.backend {
            SamplerBackend::Counter {
                nvml,
                device,
                start_mj,
            } => {
                let current_mj = nvml.get_total_energy_mj(*device)?;
                let delta_mj = current_mj.saturating_sub(*start_mj);
                let joules = delta_mj as f64 / 1000.0;
                let average_watts = joules / elapsed_secs;

                Some(EnergyMeasurement {
                    joules,
                    average_watts,
                    elapsed,
                    method: EnergyMethod::TotalEnergyCounter,
                })
            }
            SamplerBackend::Polled { joules_accum, .. } => {
                let joules = *joules_accum.lock().ok()?;
                let average_watts = joules / elapsed_secs;

                Some(EnergyMeasurement {
                    joules,
                    average_watts,
                    elapsed,
                    method: EnergyMethod::PolledPower,
                })
            }
            SamplerBackend::Unavailable => None,
        }
    }
}

impl Drop for EnergySampler {
    fn drop(&mut self) {
        if let SamplerBackend::Polled { stop_signal, .. } = &self.backend {
            stop_signal.store(true, Ordering::Relaxed);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_energy_measurement_tokens_per_joule() {
        let m = EnergyMeasurement {
            joules: 5.0,
            average_watts: 50.0,
            elapsed: Duration::from_millis(100),
            method: EnergyMethod::TotalEnergyCounter,
        };

        assert_eq!(m.tokens_per_joule(25), 5.0);
        assert_eq!(m.tokens_per_joule(0), 0.0);
    }

    #[test]
    fn test_energy_sampler_graceful_lifecycle() {
        let sampler = EnergySampler::start(0);
        thread::sleep(Duration::from_millis(20));
        let measurement = sampler.measure();

        // If running on an NVIDIA GPU machine with drivers loaded, measurement is Some;
        // if in headless/container without NVML, it degrades gracefully to None.
        if let Some(m) = measurement {
            assert!(m.elapsed >= Duration::from_millis(20));
            assert!(m.joules >= 0.0);
            assert!(m.average_watts >= 0.0);
        }
    }
}
