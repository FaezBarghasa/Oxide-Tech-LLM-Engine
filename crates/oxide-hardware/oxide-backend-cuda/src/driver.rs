//! Real CUDA Driver & Runtime API FFI bindings for memory management, stream execution, and kernel dispatch.

#![allow(
    non_camel_case_types,
    non_snake_case,
    dead_code,
    clippy::not_unsafe_ptr_arg_deref,
    clippy::borrow_as_ptr,
    clippy::uninlined_format_args
)]

use std::ffi::c_void;
use std::ptr;

pub type cudaError_t = i32;
pub type cudaStream_t = *mut c_void;
pub type cudaEvent_t = *mut c_void;

pub const CUDA_SUCCESS: cudaError_t = 0;
pub const CUDA_MEMCPY_HOST_TO_DEVICE: u32 = 1;
pub const CUDA_MEMCPY_DEVICE_TO_HOST: u32 = 2;
pub const CUDA_MEMCPY_DEVICE_TO_DEVICE: u32 = 3;

#[link(name = "cudart")]
unsafe extern "C" {
    pub fn cudaSetDevice(device: i32) -> cudaError_t;
    pub fn cudaGetDevice(device: *mut i32) -> cudaError_t;
    pub fn cudaGetDeviceCount(count: *mut i32) -> cudaError_t;
    pub fn cudaMalloc(dev_ptr: *mut *mut c_void, size: usize) -> cudaError_t;
    pub fn cudaFree(dev_ptr: *mut c_void) -> cudaError_t;
    pub fn cudaMemcpy(dst: *mut c_void, src: const_ptr, count: usize, kind: u32) -> cudaError_t;
    pub fn cudaMemcpyAsync(dst: *mut c_void, src: const_ptr, count: usize, kind: u32, stream: cudaStream_t) -> cudaError_t;
    pub fn cudaMemset(dev_ptr: *mut c_void, value: i32, count: usize) -> cudaError_t;
    pub fn cudaStreamCreate(stream: *mut cudaStream_t) -> cudaError_t;
    pub fn cudaStreamDestroy(stream: cudaStream_t) -> cudaError_t;
    pub fn cudaStreamSynchronize(stream: cudaStream_t) -> cudaError_t;
    pub fn cudaDeviceSynchronize() -> cudaError_t;
    pub fn cudaEventCreate(event: *mut cudaEvent_t) -> cudaError_t;
    pub fn cudaEventDestroy(event: cudaEvent_t) -> cudaError_t;
    pub fn cudaEventRecord(event: cudaEvent_t, stream: cudaStream_t) -> cudaError_t;
    pub fn cudaEventQuery(event: cudaEvent_t) -> cudaError_t;
    pub fn cudaEventSynchronize(event: cudaEvent_t) -> cudaError_t;
    pub fn cudaGetLastError() -> cudaError_t;
    pub fn cudaGetErrorString(error: cudaError_t) -> *const std::os::raw::c_char;
}

type const_ptr = *const c_void;

#[link(name = "oxide_cuda_kernels", kind = "static")]
unsafe extern "C" {
    pub fn launch_cuda_rmsnorm(
        output: *mut f32,
        input: *const f32,
        weight: *const f32,
        num_tokens: i32,
        hidden_dim: i32,
        eps: f32,
        stream: cudaStream_t,
    ) -> i32;

    pub fn launch_cuda_rope(
        q: *mut f32,
        k: *mut f32,
        cos_table: *const f32,
        sin_table: *const f32,
        seq_len: i32,
        num_heads: i32,
        head_dim: i32,
        stream: cudaStream_t,
    ) -> i32;

    pub fn launch_cuda_gemv_q4_0(
        y: *mut f32,
        weight_bytes: *const u8,
        x: *const f32,
        scales: *const u16,
        m: i32,
        k: i32,
        stream: cudaStream_t,
    ) -> i32;

    pub fn launch_cuda_gemv_q8_0(
        y: *mut f32,
        weight_bytes: *const i8,
        x: *const f32,
        scales: *const u16,
        m: i32,
        k: i32,
        stream: cudaStream_t,
    ) -> i32;

    pub fn launch_cuda_flash_decode(
        output: *mut f32,
        q: *const f32,
        k_cache: *const f32,
        v_cache: *const f32,
        num_heads: i32,
        head_dim: i32,
        num_kv_tokens: i32,
        sm_scale: f32,
        stream: cudaStream_t,
    ) -> i32;

    pub fn fwht_kernel_128(
        d_out: *mut f32,
        d_in: *const f32,
        num_blocks: i32,
    );
}

/// Safe wrapper around an allocated physical GPU buffer on the CUDA device.
#[derive(Debug)]
pub struct CudaDeviceBuffer {
    ptr: *mut c_void,
    size_bytes: usize,
}

// SAFETY: CUDA device buffers are thread-safe across host threads when synchronized properly.
unsafe impl Send for CudaDeviceBuffer {}
// SAFETY: CUDA device buffers are thread-safe across host threads when synchronized properly.
unsafe impl Sync for CudaDeviceBuffer {}

impl CudaDeviceBuffer {
    /// Allocates physical memory directly on the CUDA device via `cudaMalloc`.
    pub fn allocate(size_bytes: usize) -> Result<Self, String> {
        if size_bytes == 0 {
            return Ok(Self {
                ptr: ptr::null_mut(),
                size_bytes: 0,
            });
        }
        let mut d_ptr: *mut c_void = ptr::null_mut();
        // SAFETY: Calling cudaMalloc with valid pointer and size.
        let status = unsafe { cudaMalloc(&raw mut d_ptr, size_bytes) };
        if status != CUDA_SUCCESS {
            return Err(format!("cudaMalloc failed with status code {status}"));
        }
        Ok(Self {
            ptr: d_ptr,
            size_bytes,
        })
    }

    /// Asynchronously copies memory from host slice into device memory.
    pub fn copy_from_host_async<T>(&mut self, src: &[T], stream: cudaStream_t) -> Result<(), String> {
        let bytes = std::mem::size_of_val(src);
        if bytes > self.size_bytes {
            return Err("Host slice exceeds device buffer capacity".to_string());
        }
        // SAFETY: Calling cudaMemcpyAsync with valid device pointer, host pointer, size, and stream.
        let status = unsafe {
            cudaMemcpyAsync(
                self.ptr,
                src.as_ptr().cast(),
                bytes,
                CUDA_MEMCPY_HOST_TO_DEVICE,
                stream,
            )
        };
        if status != CUDA_SUCCESS {
            return Err(format!("cudaMemcpyAsync (H2D) failed with code {status}"));
        }
        Ok(())
    }

    /// Asynchronously copies memory from device buffer into host mutable slice.
    pub fn copy_to_host_async<T>(&self, dst: &mut [T], stream: cudaStream_t) -> Result<(), String> {
        let bytes = std::mem::size_of_val(dst);
        if bytes > self.size_bytes {
            return Err("Destination slice exceeds device buffer size".to_string());
        }
        // SAFETY: Calling cudaMemcpyAsync with valid destination host pointer, device pointer, and stream.
        let status = unsafe {
            cudaMemcpyAsync(
                dst.as_mut_ptr().cast(),
                self.ptr,
                bytes,
                CUDA_MEMCPY_DEVICE_TO_HOST,
                stream,
            )
        };
        if status != CUDA_SUCCESS {
            return Err(format!("cudaMemcpyAsync (D2H) failed with code {status}"));
        }
        Ok(())
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

impl Drop for CudaDeviceBuffer {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            // SAFETY: Freeing non-null device pointer allocated by cudaMalloc.
            unsafe {
                let _ = cudaFree(self.ptr);
            }
            self.ptr = ptr::null_mut();
        }
    }
}

/// Safe wrapper around a CUDA Stream.
#[derive(Debug)]
pub struct CudaStream {
    stream: cudaStream_t,
}

// SAFETY: CUDA streams can be transferred between threads.
unsafe impl Send for CudaStream {}
// SAFETY: CUDA streams can be transferred between threads.
unsafe impl Sync for CudaStream {}

impl CudaStream {
    pub fn new() -> Result<Self, String> {
        let mut stream: cudaStream_t = ptr::null_mut();
        // SAFETY: Creating a cudaStream.
        let status = unsafe { cudaStreamCreate(&raw mut stream) };
        if status != CUDA_SUCCESS {
            return Err(format!("cudaStreamCreate failed with status code {status}"));
        }
        Ok(Self { stream })
    }

    #[inline]
    #[must_use]
    pub fn raw(&self) -> cudaStream_t {
        self.stream
    }

    pub fn synchronize(&self) -> Result<(), String> {
        // SAFETY: Calling cudaStreamSynchronize.
        let status = unsafe { cudaStreamSynchronize(self.stream) };
        if status != CUDA_SUCCESS {
            return Err(format!("cudaStreamSynchronize failed with code {status}"));
        }
        Ok(())
    }
}

impl Drop for CudaStream {
    fn drop(&mut self) {
        if !self.stream.is_null() {
            // SAFETY: Destroying valid cudaStream.
            unsafe {
                let _ = cudaStreamDestroy(self.stream);
            }
            self.stream = ptr::null_mut();
        }
    }
}
