//! Real HailoRT C API dynamic driver loader.
//! Provides dynamic discovery and FFI for `libhailort.so` on Hailo-8, Hailo-8L, and Hailo-15 AI accelerators.

#![allow(non_camel_case_types, non_snake_case, dead_code)]

use std::ffi::c_void;
use std::ptr;
use std::sync::OnceLock;

pub type hailo_status = i32;
pub type hailo_vdevice = *mut c_void;
pub type hailo_hef = *mut c_void;
pub type hailo_configured_network_group = *mut c_void;

pub const HAILO_SUCCESS: hailo_status = 0;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct hailo_vdevice_params_t {
    pub device_count: u32,
    pub device_ids: *const *const std::os::raw::c_char,
    pub scheduling_algorithm: u32,
    pub group_id: *const std::os::raw::c_char,
}

type HailoCreateVDeviceFn = unsafe extern "C" fn(*const hailo_vdevice_params_t, *mut hailo_vdevice) -> hailo_status;
type HailoReleaseVDeviceFn = unsafe extern "C" fn(hailo_vdevice) -> hailo_status;
type HailoCreateHefFileFn = unsafe extern "C" fn(*mut hailo_hef, *const std::os::raw::c_char) -> hailo_status;
type HailoReleaseHefFn = unsafe extern "C" fn(hailo_hef) -> hailo_status;

#[derive(Debug, Clone, Copy)]
pub struct HailoDriverApi {
    pub create_vdevice: Option<HailoCreateVDeviceFn>,
    pub release_vdevice: Option<HailoReleaseVDeviceFn>,
    pub create_hef_file: Option<HailoCreateHefFileFn>,
    pub release_hef: Option<HailoReleaseHefFn>,
}

unsafe impl Send for HailoDriverApi {}
unsafe impl Sync for HailoDriverApi {}

static HAILO_API: OnceLock<Option<HailoDriverApi>> = OnceLock::new();

unsafe extern "C" {
    fn dlopen(filename: *const std::os::raw::c_char, flag: i32) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const std::os::raw::c_char) -> *mut c_void;
}

const RTLD_NOW: i32 = 2;

fn init_hailo_api() -> Option<HailoDriverApi> {
    let candidates = [
        c"libhailort.so",
        c"libhailort.so.4",
        c"/usr/lib/libhailort.so",
        c"/usr/local/lib/libhailort.so",
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

    // SAFETY: Loading function symbols from valid HailoRT library.
    unsafe {
        Some(HailoDriverApi {
            create_vdevice: load_sym(handle, c"hailo_create_vdevice"),
            release_vdevice: load_sym(handle, c"hailo_release_vdevice"),
            create_hef_file: load_sym(handle, c"hailo_create_hef_file"),
            release_hef: load_sym(handle, c"hailo_release_hef"),
        })
    }
}

pub fn get_hailo_api() -> Option<&'static HailoDriverApi> {
    HAILO_API.get_or_init(init_hailo_api).as_ref()
}

pub fn is_hailo_available() -> bool {
    get_hailo_api().is_some()
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
