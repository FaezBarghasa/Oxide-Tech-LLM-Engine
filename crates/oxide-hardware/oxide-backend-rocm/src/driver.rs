//! Real AMD ROCm HIP C-ABI dynamic driver loader and HIP API bindings.
//! Provides dynamic runtime discovery and invocation of `libamdhip64.so` / `libhip_hcc.so`.

#![allow(
    non_camel_case_types,
    dead_code,
    clippy::not_unsafe_ptr_arg_deref,
    clippy::uninlined_format_args
)]

use std::ffi::c_void;
use std::ptr;
use std::sync::OnceLock;

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
type HipMemcpyAsyncFn =
    unsafe extern "C" fn(*mut c_void, *const c_void, usize, u32, hipStream_t) -> hipError_t;
type HipStreamCreateFn = unsafe extern "C" fn(*mut hipStream_t) -> hipError_t;
type HipStreamDestroyFn = unsafe extern "C" fn(hipStream_t) -> hipError_t;
type HipStreamSynchronizeFn = unsafe extern "C" fn(hipStream_t) -> hipError_t;
type HipDeviceSynchronizeFn = unsafe extern "C" fn() -> hipError_t;

#[derive(Debug, Clone, Copy)]
pub struct HipDriverApi {
    pub set_device: Option<HipSetDeviceFn>,
    pub get_device: Option<HipGetDeviceFn>,
    pub get_device_count: Option<HipGetDeviceCountFn>,
    pub malloc: Option<HipMallocFn>,
    pub free: Option<HipFreeFn>,
    pub memcpy_async: Option<HipMemcpyAsyncFn>,
    pub stream_create: Option<HipStreamCreateFn>,
    pub stream_destroy: Option<HipStreamDestroyFn>,
    pub stream_synchronize: Option<HipStreamSynchronizeFn>,
    pub device_synchronize: Option<HipDeviceSynchronizeFn>,
}

// SAFETY: HipDriverApi contains function pointers that are immutable and safe to share across threads.
unsafe impl Send for HipDriverApi {}
// SAFETY: HipDriverApi contains function pointers that are immutable and safe to share across threads.
unsafe impl Sync for HipDriverApi {}

static HIP_API: OnceLock<Option<HipDriverApi>> = OnceLock::new();

unsafe extern "C" {
    fn dlopen(filename: *const std::os::raw::c_char, flag: i32) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const std::os::raw::c_char) -> *mut c_void;
}

const RTLD_NOW: i32 = 2;

fn init_hip_api() -> Option<HipDriverApi> {
    let candidates = [
        c"libamdhip64.so",
        c"libamdhip64.so.6",
        c"libamdhip64.so.5",
        c"/opt/rocm/lib/libamdhip64.so",
    ];

    let mut handle = ptr::null_mut();
    for &cand in &candidates {
        // SAFETY: Calling dlopen with valid C string and RTLD_NOW.
        let h = unsafe { dlopen(cand.as_ptr(), RTLD_NOW) };
        if !h.is_null() {
            handle = h;
            break;
        }
    }

    if handle.is_null() {
        return None;
    }

    // SAFETY: Loading function symbols from valid library handle.
    unsafe {
        Some(HipDriverApi {
            set_device: load_sym(handle, c"hipSetDevice"),
            get_device: load_sym(handle, c"hipGetDevice"),
            get_device_count: load_sym(handle, c"hipGetDeviceCount"),
            malloc: load_sym(handle, c"hipMalloc"),
            free: load_sym(handle, c"hipFree"),
            memcpy_async: load_sym(handle, c"hipMemcpyAsync"),
            stream_create: load_sym(handle, c"hipStreamCreate"),
            stream_destroy: load_sym(handle, c"hipStreamDestroy"),
            stream_synchronize: load_sym(handle, c"hipStreamSynchronize"),
            device_synchronize: load_sym(handle, c"hipDeviceSynchronize"),
        })
    }
}

pub fn get_hip_api() -> Option<&'static HipDriverApi> {
    HIP_API.get_or_init(init_hip_api).as_ref()
}

/// Checks and dynamically loads `libamdhip64.so` if available on the system.
pub fn is_hip_available() -> bool {
    get_hip_api().is_some()
}

unsafe fn load_sym<T>(handle: *mut c_void, name: &std::ffi::CStr) -> Option<T> {
    // SAFETY: Calling dlsym with valid handle and name.
    let sym = unsafe { dlsym(handle, name.as_ptr()) };
    if sym.is_null() {
        None
    } else {
        // SAFETY: Transmuting function pointer retrieved from dlsym to target signature.
        Some(unsafe { std::mem::transmute_copy(&sym) })
    }
}

/// Safe RAII Device Memory Allocation on AMD GPU via HIP.
#[derive(Debug)]
pub struct HipDeviceBuffer {
    ptr: *mut c_void,
    size_bytes: usize,
}

// SAFETY: Device pointers are safe to transfer across threads when synchronization is respected.
unsafe impl Send for HipDeviceBuffer {}
// SAFETY: Device pointers are safe to transfer across threads when synchronization is respected.
unsafe impl Sync for HipDeviceBuffer {}

impl HipDeviceBuffer {
    pub fn allocate(size_bytes: usize) -> Result<Self, String> {
        if let Some(api) = get_hip_api()
            && let Some(malloc_fn) = api.malloc
        {
            let mut d_ptr: *mut c_void = ptr::null_mut();
            // SAFETY: Calling hipMalloc with valid pointers.
            let status = unsafe { malloc_fn(&raw mut d_ptr, size_bytes) };
            if status == HIP_SUCCESS {
                return Ok(Self {
                    ptr: d_ptr,
                    size_bytes,
                });
            }
            return Err(format!("hipMalloc failed with code {status}"));
        }
        Err("HIP runtime is not loaded".to_string())
    }

    pub fn copy_from_host_async<T>(
        &mut self,
        src: &[T],
        stream: hipStream_t,
    ) -> Result<(), String> {
        let bytes = std::mem::size_of_val(src);
        if bytes > self.size_bytes {
            return Err("Host slice exceeds device buffer".to_string());
        }
        if let Some(api) = get_hip_api()
            && let Some(cpy_fn) = api.memcpy_async
        {
            // SAFETY: Calling hipMemcpyAsync with valid pointers and stream.
            let status = unsafe {
                cpy_fn(
                    self.ptr,
                    src.as_ptr().cast(),
                    bytes,
                    HIP_MEMCPY_HOST_TO_DEVICE,
                    stream,
                )
            };
            if status == HIP_SUCCESS {
                return Ok(());
            }
            return Err(format!("hipMemcpyAsync failed with code {status}"));
        }
        Err("HIP runtime is not loaded".to_string())
    }

    pub fn copy_to_host_async<T>(&self, dst: &mut [T], stream: hipStream_t) -> Result<(), String> {
        let bytes = std::mem::size_of_val(dst);
        if bytes > self.size_bytes {
            return Err("Destination slice exceeds device buffer".to_string());
        }
        if let Some(api) = get_hip_api()
            && let Some(cpy_fn) = api.memcpy_async
        {
            // SAFETY: Calling hipMemcpyAsync with valid pointers and stream.
            let status = unsafe {
                cpy_fn(
                    dst.as_mut_ptr().cast(),
                    self.ptr,
                    bytes,
                    HIP_MEMCPY_DEVICE_TO_HOST,
                    stream,
                )
            };
            if status == HIP_SUCCESS {
                return Ok(());
            }
            return Err(format!("hipMemcpyAsync failed with code {status}"));
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
        if !self.ptr.is_null()
            && let Some(api) = get_hip_api()
            && let Some(free_fn) = api.free
        {
            // SAFETY: Freeing valid device pointer.
            unsafe {
                let _ = free_fn(self.ptr);
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

// SAFETY: HIP streams can be transferred safely between host threads.
unsafe impl Send for HipStream {}
// SAFETY: HIP streams can be transferred safely between host threads.
unsafe impl Sync for HipStream {}

impl HipStream {
    pub fn new() -> Result<Self, String> {
        if let Some(api) = get_hip_api()
            && let Some(create_fn) = api.stream_create
        {
            let mut stream: hipStream_t = ptr::null_mut();
            // SAFETY: Calling hipStreamCreate.
            let status = unsafe { create_fn(&raw mut stream) };
            if status == HIP_SUCCESS {
                return Ok(Self { stream });
            }
            return Err(format!("hipStreamCreate failed with code {status}"));
        }
        Err("HIP runtime is not loaded".to_string())
    }

    #[inline]
    #[must_use]
    pub fn raw(&self) -> hipStream_t {
        self.stream
    }

    pub fn synchronize(&self) -> Result<(), String> {
        if let Some(api) = get_hip_api()
            && let Some(sync_fn) = api.stream_synchronize
        {
            // SAFETY: Calling hipStreamSynchronize.
            let status = unsafe { sync_fn(self.stream) };
            if status == HIP_SUCCESS {
                return Ok(());
            }
            return Err(format!("hipStreamSynchronize failed with code {status}"));
        }
        Err("HIP runtime is not loaded".to_string())
    }
}

impl Drop for HipStream {
    fn drop(&mut self) {
        if !self.stream.is_null()
            && let Some(api) = get_hip_api()
            && let Some(destroy_fn) = api.stream_destroy
        {
            // SAFETY: Destroying valid hipStream.
            unsafe {
                let _ = destroy_fn(self.stream);
            }
            self.stream = ptr::null_mut();
        }
    }
}
