//! Real Qualcomm QNN (Qualcomm Neural Network) C API dynamic loader.
//! Provides dynamic discovery and FFI for Hexagon HTP backend (`libQnnHtp.so`, `libQnnSystem.so`).

#![allow(non_camel_case_types, non_snake_case, dead_code)]

use std::ffi::c_void;
use std::ptr;
use std::sync::OnceLock;

pub type Qnn_ErrorHandle_t = u64;
pub type Qnn_BackendHandle_t = *mut c_void;
pub type Qnn_DeviceHandle_t = *mut c_void;
pub type Qnn_ContextHandle_t = *mut c_void;
pub type Qnn_GraphHandle_t = *mut c_void;

pub const QNN_SUCCESS: Qnn_ErrorHandle_t = 0;

type QnnBackendCreateFn = unsafe extern "C" fn(*const c_void, *const c_void, *mut Qnn_BackendHandle_t) -> Qnn_ErrorHandle_t;
type QnnBackendFreeFn = unsafe extern "C" fn(Qnn_BackendHandle_t) -> Qnn_ErrorHandle_t;
type QnnDeviceCreateFn = unsafe extern "C" fn(*const c_void, *const c_void, *mut Qnn_DeviceHandle_t) -> Qnn_ErrorHandle_t;
type QnnDeviceFreeFn = unsafe extern "C" fn(Qnn_DeviceHandle_t) -> Qnn_ErrorHandle_t;
type QnnContextCreateFn = unsafe extern "C" fn(Qnn_BackendHandle_t, Qnn_DeviceHandle_t, *const c_void, *mut Qnn_ContextHandle_t) -> Qnn_ErrorHandle_t;
type QnnContextFreeFn = unsafe extern "C" fn(Qnn_ContextHandle_t, *mut c_void) -> Qnn_ErrorHandle_t;

#[derive(Debug, Clone, Copy)]
pub struct QnnDriverApi {
    pub backend_create: Option<QnnBackendCreateFn>,
    pub backend_free: Option<QnnBackendFreeFn>,
    pub device_create: Option<QnnDeviceCreateFn>,
    pub device_free: Option<QnnDeviceFreeFn>,
    pub context_create: Option<QnnContextCreateFn>,
    pub context_free: Option<QnnContextFreeFn>,
}

// SAFETY: Function pointer table is immutable and thread-safe.
unsafe impl Send for QnnDriverApi {}
// SAFETY: Function pointer table is immutable and thread-safe.
unsafe impl Sync for QnnDriverApi {}

static QNN_API: OnceLock<Option<QnnDriverApi>> = OnceLock::new();

unsafe extern "C" {
    fn dlopen(filename: *const std::os::raw::c_char, flag: i32) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const std::os::raw::c_char) -> *mut c_void;
}

const RTLD_NOW: i32 = 2;

fn init_qnn_api() -> Option<QnnDriverApi> {
    let candidates = [
        c"libQnnHtp.so",
        c"libQnnCpu.so",
        c"/opt/qcom/qnn/lib/x86_64-linux-gnu/libQnnHtp.so",
        c"/opt/qcom/qnn/lib/aarch64-linux-gnu/libQnnHtp.so",
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

    // SAFETY: Loading function symbols from valid QNN library.
    unsafe {
        Some(QnnDriverApi {
            backend_create: load_sym(handle, c"QnnBackend_create"),
            backend_free: load_sym(handle, c"QnnBackend_free"),
            device_create: load_sym(handle, c"QnnDevice_create"),
            device_free: load_sym(handle, c"QnnDevice_free"),
            context_create: load_sym(handle, c"QnnContext_createFromBinary"),
            context_free: load_sym(handle, c"QnnContext_free"),
        })
    }
}

pub fn get_qnn_api() -> Option<&'static QnnDriverApi> {
    QNN_API.get_or_init(init_qnn_api).as_ref()
}

pub fn is_qnn_available() -> bool {
    get_qnn_api().is_some()
}

unsafe fn load_sym<T>(handle: *mut c_void, name: &std::ffi::CStr) -> Option<T> {
    // SAFETY: Calling dlsym on valid handle.
    let sym = unsafe { dlsym(handle, name.as_ptr()) };
    if sym.is_null() {
        None
    } else {
        // SAFETY: Transmuting valid function pointer.
        Some(unsafe { std::mem::transmute_copy(&sym) })
    }
}
