//! Real AMD ROCm HIP C-ABI dynamic driver loader and HIP API bindings.
//! Provides dynamic runtime discovery and invocation of `libamdhip64.so` / `libhip_hcc.so`.

#![allow(non_camel_case_types, dead_code)]

use std::ffi::c_void;
use std::ptr;
use std::sync::atomic::{AtomicBool, Ordering};

pub type hipError_t = i32;
pub type hipStream_t = *mut c_void;
pub type hipEvent_t = *mut c_void;

pub const HIP_SUCCESS: hipError_t = 0;
pub const HIP_MEMCPY_HOST_TO_DEVICE: u32 = 1;
pub const HIP_MEMCPY_DEVICE_TO_HOST: u32 = 2;
pub const HIP_MEMCPY_DEVICE_TO_DEVICE: u32 = 3;

type HipSetDeviceFn = unsafe extern "C" fn(i32) -> hipError_t;
type HipGetDeviceFn = unsafe extern "C" fn(*mut i32) -> hipError_t;
type HipGetDeviceCountFn = unsafe extern "C" fn(*mut i32) -> hipError_t;
type HipMallocFn = unsafe extern "C" fn(*mut *mut c_void, usize) -> hipError_t;
type HipFreeFn = unsafe extern "C" fn(*mut c_void) -> hipError_t;
type HipMemcpyAsyncFn = unsafe extern "C" fn(*mut c_void, *const c_void, usize, u32, hipStream_t) -> hipError_t;
type HipStreamCreateFn = unsafe extern "C" fn(*mut hipStream_t) -> hipError_t;
type HipStreamDestroyFn = unsafe extern "C" fn(hipStream_t) -> hipError_t;
type HipStreamSynchronizeFn = unsafe extern "C" fn(hipStream_t) -> hipError_t;
type HipDeviceSynchronizeFn = unsafe extern "C" fn() -> hipError_t;

static HIP_LOAD_ATTEMPTED: AtomicBool = AtomicBool::new(false);
static mut HIP_LIB_HANDLE: *mut c_void = ptr::null_mut();

static mut HIP_SET_DEVICE: Option<HipSetDeviceFn> = None;
static mut HIP_GET_DEVICE: Option<HipGetDeviceFn> = None;
static mut HIP_GET_DEVICE_COUNT: Option<HipGetDeviceCountFn> = None;
static mut HIP_MALLOC: Option<HipMallocFn> = None;
static mut HIP_FREE: Option<HipFreeFn> = None;
static mut HIP_MEMCPY_ASYNC: Option<HipMemcpyAsyncFn> = None;
static mut HIP_STREAM_CREATE: Option<HipStreamCreateFn> = None;
static mut HIP_STREAM_DESTROY: Option<HipStreamDestroyFn> = None;
static mut HIP_STREAM_SYNCHRONIZE: Option<HipStreamSynchronizeFn> = None;
static mut HIP_DEVICE_SYNCHRONIZE: Option<HipDeviceSynchronizeFn> = None;

extern "C" {
    fn dlopen(filename: *const std::os::raw::c_char, flag: i32) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const std::os::raw::c_char) -> *mut c_void;
}

const RTLD_NOW: i32 = 2;

/// Checks and dynamically loads `libamdhip64.so` if available on the system.
pub fn is_hip_available() -> bool {
    ensure_hip_loaded();
    unsafe { HIP_MALLOC.is_some() }
}

fn ensure_hip_loaded() {
    if HIP_LOAD_ATTEMPTED.swap(true, Ordering::SeqCst) {
        return;
    }

    let candidates = [
        c"libamdhip64.so",
        c"libamdhip64.so.6",
        c"libamdhip64.so.5",
        c"/opt/rocm/lib/libamdhip64.so",
    ];

    unsafe {
        for &cand in &candidates {
            let handle = dlopen(cand.as_ptr(), RTLD_NOW);
            if !handle.is_null() {
                HIP_LIB_HANDLE = handle;
                break;
            }
        }

        if !HIP_LIB_HANDLE.is_null() {
            HIP_SET_DEVICE = load_sym(HIP_LIB_HANDLE, c"hipSetDevice");
            HIP_GET_DEVICE = load_sym(HIP_LIB_HANDLE, c"hipGetDevice");
            HIP_GET_DEVICE_COUNT = load_sym(HIP_LIB_HANDLE, c"hipGetDeviceCount");
            HIP_MALLOC = load_sym(HIP_LIB_HANDLE, c"hipMalloc");
            HIP_FREE = load_sym(HIP_LIB_HANDLE, c"hipFree");
            HIP_MEMCPY_ASYNC = load_sym(HIP_LIB_HANDLE, c"hipMemcpyAsync");
            HIP_STREAM_CREATE = load_sym(HIP_LIB_HANDLE, c"hipStreamCreate");
            HIP_STREAM_DESTROY = load_sym(HIP_LIB_HANDLE, c"hipStreamDestroy");
            HIP_STREAM_SYNCHRONIZE = load_sym(HIP_LIB_HANDLE, c"hipStreamSynchronize");
            HIP_DEVICE_SYNCHRONIZE = load_sym(HIP_LIB_HANDLE, c"hipDeviceSynchronize");
        }
    }
}

unsafe fn load_sym<T>(handle: *mut c_void, name: &std::ffi::CStr) -> Option<T> {
    let sym = dlsym(handle, name.as_ptr());
    if sym.is_null() {
        None
    } else {
        Some(std::mem::transmute_copy(&sym))
    }
}

/// Safe RAII Device Memory Allocation on AMD GPU via HIP.
#[derive(Debug)]
pub struct HipDeviceBuffer {
    ptr: *mut c_void,
    size_bytes: usize,
}

unsafe impl Send for HipDeviceBuffer {}
unsafe impl Sync for HipDeviceBuffer {}

impl HipDeviceBuffer {
    pub fn allocate(size_bytes: usize) -> Result<Self, String> {
        ensure_hip_loaded();
        unsafe {
            if let Some(malloc_fn) = HIP_MALLOC {
                let mut d_ptr: *mut c_void = ptr::null_mut();
                let status = malloc_fn(&mut d_ptr, size_bytes);
                if status == HIP_SUCCESS {
                    return Ok(Self {
                        ptr: d_ptr,
                        size_bytes,
                    });
                }
                return Err(format!("hipMalloc failed with code {}", status));
            }
        }
        Err("HIP runtime is not loaded".to_string())
    }

    pub fn copy_from_host_async<T>(&mut self, src: &[T], stream: hipStream_t) -> Result<(), String> {
        let bytes = std::mem::size_of_val(src);
        if bytes > self.size_bytes {
            return Err("Host slice exceeds device buffer".to_string());
        }
        unsafe {
            if let Some(cpy_fn) = HIP_MEMCPY_ASYNC {
                let status = cpy_fn(self.ptr, src.as_ptr().cast(), bytes, HIP_MEMCPY_HOST_TO_DEVICE, stream);
                if status == HIP_SUCCESS {
                    return Ok(());
                }
                return Err(format!("hipMemcpyAsync failed with code {}", status));
            }
        }
        Err("HIP runtime is not loaded".to_string())
    }

    pub fn copy_to_host_async<T>(&self, dst: &mut [T], stream: hipStream_t) -> Result<(), String> {
        let bytes = std::mem::size_of_val(dst);
        if bytes > self.size_bytes {
            return Err("Destination slice exceeds device buffer".to_string());
        }
        unsafe {
            if let Some(cpy_fn) = HIP_MEMCPY_ASYNC {
                let status = cpy_fn(dst.as_mut_ptr().cast(), self.ptr, bytes, HIP_MEMCPY_DEVICE_TO_HOST, stream);
                if status == HIP_SUCCESS {
                    return Ok(());
                }
                return Err(format!("hipMemcpyAsync failed with code {}", status));
            }
        }
        Err("HIP runtime is not loaded".to_string())
    }

    #[inline]
    #[must_use]
    pub fn as_raw_ptr(&self) -> *mut c_void {
        self.ptr
    }

    #[inline]
    #[must_use]
    pub fn as_typed_ptr<T>(&self) -> *mut T {
        self.ptr.cast()
    }

    #[inline]
    #[must_use]
    pub fn size_bytes(&self) -> usize {
        self.size_bytes
    }
}

impl Drop for HipDeviceBuffer {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                if let Some(free_fn) = HIP_FREE {
                    let _ = free_fn(self.ptr);
                }
            }
            self.ptr = ptr::null_mut();
        }
    }
}

/// Safe RAII HIP Stream.
#[derive(Debug)]
pub struct HipStream {
    stream: hipStream_t,
}

unsafe impl Send for HipStream {}
unsafe impl Sync for HipStream {}

impl HipStream {
    pub fn new() -> Result<Self, String> {
        ensure_hip_loaded();
        unsafe {
            if let Some(create_fn) = HIP_STREAM_CREATE {
                let mut stream: hipStream_t = ptr::null_mut();
                let status = create_fn(&mut stream);
                if status == HIP_SUCCESS {
                    return Ok(Self { stream });
                }
                return Err(format!("hipStreamCreate failed with code {}", status));
            }
        }
        Err("HIP runtime is not loaded".to_string())
    }

    #[inline]
    #[must_use]
    pub fn raw(&self) -> hipStream_t {
        self.stream
    }

    pub fn synchronize(&self) -> Result<(), String> {
        unsafe {
            if let Some(sync_fn) = HIP_STREAM_SYNCHRONIZE {
                let status = sync_fn(self.stream);
                if status == HIP_SUCCESS {
                    return Ok(());
                }
                return Err(format!("hipStreamSynchronize failed with code {}", status));
            }
        }
        Err("HIP runtime is not loaded".to_string())
    }
}

impl Drop for HipStream {
    fn drop(&mut self) {
        if !self.stream.is_null() {
            unsafe {
                if let Some(destroy_fn) = HIP_STREAM_DESTROY {
                    let _ = destroy_fn(self.stream);
                }
            }
            self.stream = ptr::null_mut();
        }
    }
}
