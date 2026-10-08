//! Real Google TPU runtime dynamic driver loader.
//! Supports Google Coral Edge TPU (`libedgetpu.so.1`) and Google Cloud TPU PJRT C API (`libpjrt_c_api.so`).

#![allow(non_camel_case_types, non_snake_case, dead_code)]

use std::ffi::c_void;
use std::ptr;
use std::sync::OnceLock;

// Google Coral Edge TPU C API types
pub type edgetpu_device_type = i32;
pub const EDGETPU_APEX_PCI: edgetpu_device_type = 0;
pub const EDGETPU_APEX_USB: edgetpu_device_type = 1;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct edgetpu_device {
    pub type_: edgetpu_device_type,
    pub path: *const std::os::raw::c_char,
}

type EdgeTpuListDevicesFn = unsafe extern "C" fn(*mut usize) -> *mut edgetpu_device;
type EdgeTpuFreeDevicesFn = unsafe extern "C" fn(*mut edgetpu_device);
type EdgeTpuCreateDeviceFn = unsafe extern "C" fn(edgetpu_device_type, *const std::os::raw::c_char) -> *mut c_void;

#[derive(Debug, Clone, Copy)]
pub struct EdgeTpuDriverApi {
    pub list_devices: Option<EdgeTpuListDevicesFn>,
    pub free_devices: Option<EdgeTpuFreeDevicesFn>,
    pub create_device: Option<EdgeTpuCreateDeviceFn>,
}

// SAFETY: Function pointer table is immutable and thread-safe.
unsafe impl Send for EdgeTpuDriverApi {}
// SAFETY: Function pointer table is immutable and thread-safe.
unsafe impl Sync for EdgeTpuDriverApi {}

static EDGETPU_API: OnceLock<Option<EdgeTpuDriverApi>> = OnceLock::new();

// Google Cloud TPU PJRT C API types
pub type PJRT_Error = *mut c_void;
pub type PJRT_Client = *mut c_void;

type PjrtClientCreateFn = unsafe extern "C" fn(*mut PJRT_Client) -> PJRT_Error;

#[derive(Debug, Clone, Copy)]
pub struct PjrtTpuDriverApi {
    pub client_create: Option<PjrtClientCreateFn>,
}

// SAFETY: Function pointer table is immutable and thread-safe.
unsafe impl Send for PjrtTpuDriverApi {}
// SAFETY: Function pointer table is immutable and thread-safe.
unsafe impl Sync for PjrtTpuDriverApi {}

static PJRT_API: OnceLock<Option<PjrtTpuDriverApi>> = OnceLock::new();

unsafe extern "C" {
    fn dlopen(filename: *const std::os::raw::c_char, flag: i32) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const std::os::raw::c_char) -> *mut c_void;
}

const RTLD_NOW: i32 = 2;

fn init_edgetpu_api() -> Option<EdgeTpuDriverApi> {
    let candidates = [
        c"libedgetpu.so.1",
        c"libedgetpu.so.1.0",
        c"libedgetpu.so",
        c"/usr/lib/x86_64-linux-gnu/libedgetpu.so.1",
    ];

    let mut handle = ptr::null_mut();
    for &cand in &candidates {
        // SAFETY: Calling dlopen with valid C string.
        let h = unsafe { dlopen(cand.as_ptr(), RTLD_NOW) };
        if !h.is_null() {
            handle = h;
            break;
        }
    }

    if handle.is_null() {
        return None;
    }

    // SAFETY: Loading function symbols from valid Edge TPU library.
    unsafe {
        Some(EdgeTpuDriverApi {
            list_devices: load_sym(handle, c"edgetpu_list_devices"),
            free_devices: load_sym(handle, c"edgetpu_free_devices"),
            create_device: load_sym(handle, c"edgetpu_create_device"),
        })
    }
}

fn init_pjrt_api() -> Option<PjrtTpuDriverApi> {
    let candidates = [
        c"libpjrt_c_api.so",
        c"libpjrt_tpu.so",
        c"/usr/lib/libpjrt_c_api.so",
    ];

    let mut handle = ptr::null_mut();
    for &cand in &candidates {
        // SAFETY: Calling dlopen with valid C string.
        let h = unsafe { dlopen(cand.as_ptr(), RTLD_NOW) };
        if !h.is_null() {
            handle = h;
            break;
        }
    }

    if handle.is_null() {
        return None;
    }

    // SAFETY: Loading function symbols from valid PJRT library.
    unsafe {
        Some(PjrtTpuDriverApi {
            client_create: load_sym(handle, c"PJRT_Client_Create"),
        })
    }
}

pub fn get_edgetpu_api() -> Option<&'static EdgeTpuDriverApi> {
    EDGETPU_API.get_or_init(init_edgetpu_api).as_ref()
}

pub fn is_edgetpu_available() -> bool {
    get_edgetpu_api().is_some()
}

pub fn get_pjrt_api() -> Option<&'static PjrtTpuDriverApi> {
    PJRT_API.get_or_init(init_pjrt_api).as_ref()
}

pub fn is_cloud_tpu_available() -> bool {
    get_pjrt_api().is_some()
}

unsafe fn load_sym<T>(handle: *mut c_void, name: &std::ffi::CStr) -> Option<T> {
    // SAFETY: Calling dlsym with valid handle.
    let sym = unsafe { dlsym(handle, name.as_ptr()) };
    if sym.is_null() {
        None
    } else {
        // SAFETY: Transmuting valid function pointer.
        Some(unsafe { std::mem::transmute_copy(&sym) })
    }
}
