//! Real Intel oneAPI Level-Zero Core API dynamic driver loader and memory manager.
//! Dynamically loads `libze_loader.so.1` / `libze_loader.so` (`zeInit`, `zeDriverGet`, `zeDeviceGet`, `zeMemAllocDevice`, `zeMemFree`).

#![allow(non_camel_case_types, non_snake_case, dead_code)]

use std::ffi::c_void;
use std::ptr;
use std::sync::OnceLock;

pub type ze_result_t = i32;
pub type ze_driver_handle_t = *mut c_void;
pub type ze_device_handle_t = *mut c_void;
pub type ze_context_handle_t = *mut c_void;
pub type ze_command_queue_handle_t = *mut c_void;

pub const ZE_RESULT_SUCCESS: ze_result_t = 0;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ze_init_flags_t(pub u32);
pub const ZE_INIT_FLAG_GPU_ONLY: ze_init_flags_t = ze_init_flags_t(1);

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ze_device_mem_alloc_desc_t {
    pub stype: u32,
    pub pNext: *const c_void,
    pub flags: u32,
    pub ordinal: u32,
}

type ZeInitFn = unsafe extern "C" fn(ze_init_flags_t) -> ze_result_t;
type ZeDriverGetFn = unsafe extern "C" fn(*mut u32, *mut ze_driver_handle_t) -> ze_result_t;
type ZeDeviceGetFn =
    unsafe extern "C" fn(ze_driver_handle_t, *mut u32, *mut ze_device_handle_t) -> ze_result_t;
type ZeContextCreateFn = unsafe extern "C" fn(
    ze_driver_handle_t,
    *const c_void,
    *mut ze_context_handle_t,
) -> ze_result_t;
type ZeContextDestroyFn = unsafe extern "C" fn(ze_context_handle_t) -> ze_result_t;
type ZeMemAllocDeviceFn = unsafe extern "C" fn(
    ze_context_handle_t,
    *const ze_device_mem_alloc_desc_t,
    usize,
    usize,
    ze_device_handle_t,
    *mut *mut c_void,
) -> ze_result_t;
type ZeMemFreeFn = unsafe extern "C" fn(ze_context_handle_t, *mut c_void) -> ze_result_t;

#[derive(Debug, Clone, Copy)]
pub struct LevelZeroDriverApi {
    pub init: Option<ZeInitFn>,
    pub driver_get: Option<ZeDriverGetFn>,
    pub device_get: Option<ZeDeviceGetFn>,
    pub context_create: Option<ZeContextCreateFn>,
    pub context_destroy: Option<ZeContextDestroyFn>,
    pub mem_alloc_device: Option<ZeMemAllocDeviceFn>,
    pub mem_free: Option<ZeMemFreeFn>,
}

// SAFETY: Function pointer table is immutable and thread-safe.
unsafe impl Send for LevelZeroDriverApi {}
// SAFETY: Function pointer table is immutable and thread-safe.
unsafe impl Sync for LevelZeroDriverApi {}

static ZE_API: OnceLock<Option<LevelZeroDriverApi>> = OnceLock::new();

unsafe extern "C" {
    fn dlopen(filename: *const std::os::raw::c_char, flag: i32) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const std::os::raw::c_char) -> *mut c_void;
}

const RTLD_NOW: i32 = 2;

fn init_ze_api() -> Option<LevelZeroDriverApi> {
    let candidates = [
        c"libze_loader.so.1",
        c"libze_loader.so",
        c"/usr/lib/x86_64-linux-gnu/libze_loader.so.1",
        c"/usr/local/lib/libze_loader.so.1",
    ];

    let mut handle = ptr::null_mut();
    for &cand in &candidates {
        // SAFETY: Calling dlopen with candidate library path.
        let h = unsafe { dlopen(cand.as_ptr(), RTLD_NOW) };
        if !h.is_null() {
            handle = h;
            break;
        }
    }

    if handle.is_null() {
        return None;
    }

    // SAFETY: Loading Level-Zero function pointers from opened library handle.
    unsafe {
        Some(LevelZeroDriverApi {
            init: load_sym(handle, c"zeInit"),
            driver_get: load_sym(handle, c"zeDriverGet"),
            device_get: load_sym(handle, c"zeDeviceGet"),
            context_create: load_sym(handle, c"zeContextCreate"),
            context_destroy: load_sym(handle, c"zeContextDestroy"),
            mem_alloc_device: load_sym(handle, c"zeMemAllocDevice"),
            mem_free: load_sym(handle, c"zeMemFree"),
        })
    }
}

pub fn get_ze_api() -> Option<&'static LevelZeroDriverApi> {
    ZE_API.get_or_init(init_ze_api).as_ref()
}

pub fn is_level_zero_available() -> bool {
    get_ze_api().is_some()
}

unsafe fn load_sym<T>(handle: *mut c_void, name: &std::ffi::CStr) -> Option<T> {
    // SAFETY: Calling dlsym on valid handle with valid CStr.
    let sym = unsafe { dlsym(handle, name.as_ptr()) };
    if sym.is_null() {
        None
    } else {
        // SAFETY: Transmuting loaded symbol pointer to target function pointer type.
        Some(unsafe { std::mem::transmute_copy(&sym) })
    }
}

/// Safe RAII Device Memory Allocation on Intel GPU via oneAPI Level-Zero.
#[derive(Debug)]
pub struct LevelZeroDeviceBuffer {
    ptr: *mut c_void,
    size_bytes: usize,
    context: ze_context_handle_t,
}

// SAFETY: Device pointer is exclusively managed and deallocated via Level-Zero context.
unsafe impl Send for LevelZeroDeviceBuffer {}
// SAFETY: Device pointer is exclusively managed and deallocated via Level-Zero context.
unsafe impl Sync for LevelZeroDeviceBuffer {}

impl LevelZeroDeviceBuffer {
    /// Allocates physical memory on an Intel GPU device.
    ///
    /// # Safety
    /// `context` and `device` must be valid, initialized Level-Zero handles.
    pub unsafe fn allocate(
        context: ze_context_handle_t,
        device: ze_device_handle_t,
        size_bytes: usize,
    ) -> Result<Self, String> {
        if let Some(api) = get_ze_api()
            && let Some(alloc_fn) = api.mem_alloc_device
        {
            let desc = ze_device_mem_alloc_desc_t {
                stype: 0x10006, // ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC
                pNext: ptr::null(),
                flags: 0,
                ordinal: 0,
            };
            let mut d_ptr: *mut c_void = ptr::null_mut();
            // SAFETY: Calling zeMemAllocDevice with valid descriptors and handles.
            let status = unsafe {
                alloc_fn(
                    context,
                    &raw const desc,
                    size_bytes,
                    64,
                    device,
                    &raw mut d_ptr,
                )
            };
            if status == ZE_RESULT_SUCCESS {
                return Ok(Self {
                    ptr: d_ptr,
                    size_bytes,
                    context,
                });
            }
            return Err(format!("zeMemAllocDevice failed with code {status}"));
        }
        Err("Intel Level-Zero driver is not available".to_string())
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

impl Drop for LevelZeroDeviceBuffer {
    fn drop(&mut self) {
        if !self.ptr.is_null()
            && let Some(api) = get_ze_api()
            && let Some(free_fn) = api.mem_free
        {
            // SAFETY: Freeing device buffer with matching context.
            unsafe {
                let _ = free_fn(self.context, self.ptr);
            }
            self.ptr = ptr::null_mut();
        }
    }
}
