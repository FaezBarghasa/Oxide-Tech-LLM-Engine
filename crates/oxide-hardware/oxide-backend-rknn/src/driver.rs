//! Real Rockchip RKNN Runtime C ABI dynamic driver loader.
//! Provides dynamic discovery and FFI for `librknnrt.so` / `librknn_api.so` on RK3588/RK3576/RV1109/RV1106.

#![allow(non_camel_case_types, non_snake_case, dead_code)]

use std::ffi::c_void;
use std::ptr;
use std::sync::OnceLock;

pub type rknn_context = u64;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct rknn_input {
    pub index: u32,
    pub buf: *mut c_void,
    pub size: u32,
    pub pass_through: u8,
    pub type_: u32,
    pub fmt: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct rknn_output {
    pub want_float: u8,
    pub is_prealloc: u8,
    pub index: u32,
    pub buf: *mut c_void,
    pub size: u32,
}

type RknnInitFn =
    unsafe extern "C" fn(*mut rknn_context, *mut c_void, u32, u32, *mut c_void) -> i32;
type RknnDestroyFn = unsafe extern "C" fn(rknn_context) -> i32;
type RknnInputsSetFn = unsafe extern "C" fn(rknn_context, u32, *mut rknn_input) -> i32;
type RknnRunFn = unsafe extern "C" fn(rknn_context, *mut c_void) -> i32;
type RknnOutputsGetFn =
    unsafe extern "C" fn(rknn_context, u32, *mut rknn_output, *mut c_void) -> i32;
type RknnOutputsReleaseFn = unsafe extern "C" fn(rknn_context, u32, *mut rknn_output) -> i32;

#[derive(Debug, Clone, Copy)]
pub struct RknnDriverApi {
    pub init: Option<RknnInitFn>,
    pub destroy: Option<RknnDestroyFn>,
    pub inputs_set: Option<RknnInputsSetFn>,
    pub run: Option<RknnRunFn>,
    pub outputs_get: Option<RknnOutputsGetFn>,
    pub outputs_release: Option<RknnOutputsReleaseFn>,
}

// SAFETY: Function pointer table is immutable and thread-safe.
unsafe impl Send for RknnDriverApi {}
// SAFETY: Function pointer table is immutable and thread-safe.
unsafe impl Sync for RknnDriverApi {}

static RKNN_API: OnceLock<Option<RknnDriverApi>> = OnceLock::new();

unsafe extern "C" {
    fn dlopen(filename: *const std::os::raw::c_char, flag: i32) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const std::os::raw::c_char) -> *mut c_void;
}

const RTLD_NOW: i32 = 2;

fn init_rknn_api() -> Option<RknnDriverApi> {
    let candidates = [
        c"librknnrt.so",
        c"librknn_api.so",
        c"/usr/lib/librknnrt.so",
        c"/usr/local/lib/librknnrt.so",
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

    // SAFETY: Loading function symbols from valid RKNN library.
    unsafe {
        Some(RknnDriverApi {
            init: load_sym(handle, c"rknn_init"),
            destroy: load_sym(handle, c"rknn_destroy"),
            inputs_set: load_sym(handle, c"rknn_inputs_set"),
            run: load_sym(handle, c"rknn_run"),
            outputs_get: load_sym(handle, c"rknn_outputs_get"),
            outputs_release: load_sym(handle, c"rknn_outputs_release"),
        })
    }
}

pub fn get_rknn_api() -> Option<&'static RknnDriverApi> {
    RKNN_API.get_or_init(init_rknn_api).as_ref()
}

pub fn is_rknn_available() -> bool {
    get_rknn_api().is_some()
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
