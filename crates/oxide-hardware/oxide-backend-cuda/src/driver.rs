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
    pub fn cudaMemcpyAsync(
        dst: *mut c_void,
        src: const_ptr,
        count: usize,
        kind: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub fn cudaMemset(dev_ptr: *mut c_void, value: i32, count: usize) -> cudaError_t;
    pub fn cudaMemsetAsync(
        dev_ptr: *mut c_void,
        value: i32,
        count: usize,
        stream: cudaStream_t,
    ) -> cudaError_t;
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
    pub fn cudaStreamBeginCapture(stream: cudaStream_t, mode: u32) -> cudaError_t;
    pub fn cudaStreamEndCapture(stream: cudaStream_t, pGraph: *mut cudaGraph_t) -> cudaError_t;
    pub fn cudaGraphInstantiate(
        pGraphExec: *mut cudaGraphExec_t,
        graph: cudaGraph_t,
        flags: u64,
    ) -> cudaError_t;
    pub fn cudaGraphLaunch(graphExec: cudaGraphExec_t, stream: cudaStream_t) -> cudaError_t;
    pub fn cudaGraphExecDestroy(graphExec: cudaGraphExec_t) -> cudaError_t;
    pub fn cudaGraphDestroy(graph: cudaGraph_t) -> cudaError_t;
}

pub type cudaGraph_t = *mut c_void;
pub type cudaGraphExec_t = *mut c_void;

pub const CUDA_STREAM_CAPTURE_MODE_GLOBAL: u32 = 0;
pub const CUDA_STREAM_CAPTURE_MODE_THREAD_LOCAL: u32 = 1;
pub const CUDA_STREAM_CAPTURE_MODE_RELAXED: u32 = 2;

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
        weight_offset: f32,
        stream: cudaStream_t,
    ) -> i32;

    pub fn launch_cuda_fused_rmsnorm_q8_1(
        output: *mut f32,
        y_q8_1: *mut std::ffi::c_void,
        input: *const f32,
        weight: *const f32,
        num_tokens: i32,
        hidden_dim: i32,
        eps: f32,
        weight_offset: f32,
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
        raw_blocks: *const std::ffi::c_void,
        x: *const f32,
        m: i32,
        k: i32,
        stream: cudaStream_t,
    ) -> i32;

    pub fn launch_cuda_gemv_q5_0(
        y: *mut f32,
        raw_blocks: *const std::ffi::c_void,
        x: *const f32,
        m: i32,
        k: i32,
        stream: cudaStream_t,
    ) -> i32;

    pub fn launch_cuda_gemv_tq2_0(
        y: *mut f32,
        raw_blocks: *const std::ffi::c_void,
        x: *const f32,
        m: i32,
        k: i32,
        stream: cudaStream_t,
    ) -> i32;

    pub fn launch_cuda_quantize_q8_1(
        x: *const f32,
        y_q8_1: *mut std::ffi::c_void,
        n: i32,
        stream: cudaStream_t,
    ) -> i32;

    pub fn launch_cuda_mmvq_q8_0(
        y: *mut f32,
        raw_blocks: *const std::ffi::c_void,
        x: *const std::ffi::c_void,
        m: i32,
        k: i32,
        batch_size: i32,
        stream: cudaStream_t,
    ) -> i32;

    pub fn launch_cuda_gemv_q8_0(
        y: *mut f32,
        raw_blocks: *const std::ffi::c_void,
        x: *const std::ffi::c_void,
        m: i32,
        k: i32,
        stream: cudaStream_t,
    ) -> i32;

    pub fn launch_cuda_mmvq_q6_k(
        y: *mut f32,
        raw_blocks: *const std::ffi::c_void,
        x: *const std::ffi::c_void,
        m: i32,
        k: i32,
        batch_size: i32,
        stream: cudaStream_t,
    ) -> i32;

    pub fn launch_cuda_gemv_q6_k(
        y: *mut f32,
        raw_blocks: *const std::ffi::c_void,
        x: *const std::ffi::c_void,
        m: i32,
        k: i32,
        stream: cudaStream_t,
    ) -> i32;

    pub fn launch_cuda_mmvq_q4_k(
        y: *mut f32,
        raw_blocks: *const std::ffi::c_void,
        x: *const std::ffi::c_void,
        m: i32,
        k: i32,
        batch_size: i32,
        stream: cudaStream_t,
    ) -> i32;

    pub fn launch_cuda_gemv_q4_k(
        y: *mut f32,
        raw_blocks: *const std::ffi::c_void,
        x: *const std::ffi::c_void,
        m: i32,
        k: i32,
        stream: cudaStream_t,
    ) -> i32;

    pub fn launch_cuda_gemv_ptq1_0(
        y: *mut f32,
        raw_blocks: *const std::ffi::c_void,
        x: *const f32,
        m: i32,
        k: i32,
        stream: cudaStream_t,
    ) -> i32;

    pub fn launch_cuda_gemv_q2_k(
        y: *mut f32,
        raw_blocks: *const std::ffi::c_void,
        x: *const f32,
        m: i32,
        k: i32,
        stream: cudaStream_t,
    ) -> i32;

    pub fn launch_cuda_gemv_q3_k(
        y: *mut f32,
        raw_blocks: *const std::ffi::c_void,
        x: *const f32,
        m: i32,
        k: i32,
        stream: cudaStream_t,
    ) -> i32;

    pub fn launch_cuda_gemv_q5_k(
        y: *mut f32,
        raw_blocks: *const std::ffi::c_void,
        x: *const f32,
        m: i32,
        k: i32,
        stream: cudaStream_t,
    ) -> i32;

    pub fn launch_cuda_gemv_q4_1(
        y: *mut f32,
        raw_blocks: *const std::ffi::c_void,
        x: *const f32,
        m: i32,
        k: i32,
        stream: cudaStream_t,
    ) -> i32;

    pub fn launch_cuda_gemv_q5_1(
        y: *mut f32,
        raw_blocks: *const std::ffi::c_void,
        x: *const f32,
        m: i32,
        k: i32,
        stream: cudaStream_t,
    ) -> i32;

    pub fn launch_cuda_gemv_iq4_nl(
        y: *mut f32,
        raw_blocks: *const std::ffi::c_void,
        x: *const f32,
        m: i32,
        k: i32,
        stream: cudaStream_t,
    ) -> i32;

    pub fn launch_cuda_vector_add(a: *mut f32, b: *const f32, n: i32, stream: cudaStream_t) -> i32;
    pub fn launch_cuda_vector_scale(x: *mut f32, scale: f32, n: i32, stream: cudaStream_t) -> i32;

    pub fn launch_cuda_swiglu(
        out: *mut f32,
        gate: *const f32,
        up: *const f32,
        n: i32,
        stream: cudaStream_t,
    ) -> i32;

    pub fn launch_cuda_fused_swiglu_q8_1(
        y_q8_1: *mut std::ffi::c_void,
        out_f32: *mut f32,
        gate: *const f32,
        up: *const f32,
        n: i32,
        stream: cudaStream_t,
    ) -> i32;

    pub fn launch_cuda_rope_single(
        q: *mut f32,
        k: *mut f32,
        k_out: *mut f32,
        pos: i32,
        num_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        theta: f32,
        stream: cudaStream_t,
    ) -> i32;

    pub fn launch_cuda_flash_decode(
        output: *mut f32,
        q: *const f32,
        k_cache: *const f32,
        v_cache: *const f32,
        num_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        num_kv_tokens: i32,
        start_token: i32,
        sm_scale: f32,
        stream: cudaStream_t,
    ) -> i32;

    pub fn launch_cuda_rope_batch(
        q: *mut f32,
        k: *mut f32,
        k_cache: *mut f32,
        start_pos: i32,
        batch_size: i32,
        num_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        theta: f32,
        stream: cudaStream_t,
    ) -> i32;

    pub fn launch_cuda_flash_causal_prefill(
        output: *mut f32,
        q: *const f32,
        k_cache: *const f32,
        v_cache: *const f32,
        start_pos: i32,
        batch_size: i32,
        num_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        kv_cap: i32,
        sm_scale: f32,
        stream: cudaStream_t,
    ) -> i32;


    pub fn launch_cuda_qk_norm(
        qk: *mut f32,
        weight: *const f32,
        num_heads: i32,
        head_dim: i32,
        eps: f32,
        stream: cudaStream_t,
    ) -> i32;

    pub fn launch_cuda_gelu(
        out: *mut f32,
        gate: *const f32,
        up: *const f32,
        n: i32,
        stream: cudaStream_t,
    ) -> i32;

    pub fn launch_cuda_logit_softcap(
        logits: *mut f32,
        vocab_size: i32,
        cap: f32,
        stream: cudaStream_t,
    ) -> i32;

    pub fn launch_cuda_argmax(
        logits: *const f32,
        vocab_size: i32,
        out_token: *mut u32,
        out_max_val: *mut f32,
        batch_size: i32,
        stream: cudaStream_t,
    ) -> i32;

    pub fn fwht_kernel_128(d_out: *mut f32, d_in: *const f32, num_blocks: i32);
    pub fn init_cuda_kernel_attributes();
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
    /// Creates an empty device buffer pointing to null.
    #[must_use]
    pub fn empty() -> Self {
        Self {
            ptr: ptr::null_mut(),
            size_bytes: 0,
        }
    }

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
            return Err(format!(
                "cudaMalloc failed with status code {status} (requested {size_bytes} bytes = {:.2} MB)",
                size_bytes as f64 / 1_048_576.0
            ));
        }
        Ok(Self {
            ptr: d_ptr,
            size_bytes,
        })
    }

    /// Asynchronously copies memory from host slice into device memory.
    pub fn copy_from_host_async<T>(
        &mut self,
        src: &[T],
        stream: cudaStream_t,
    ) -> Result<(), String> {
        let bytes = std::mem::size_of_val(src);
        if bytes == 0 {
            return Ok(());
        }
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
        if bytes == 0 {
            return Ok(());
        }
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

    pub fn memset_zero(&mut self) -> Result<(), String> {
        if !self.ptr.is_null() && self.size_bytes > 0 {
            let status = unsafe { cudaMemset(self.ptr, 0, self.size_bytes) };
            if status != CUDA_SUCCESS {
                return Err(format!("cudaMemset failed: {status}"));
            }
        }
        Ok(())
    }

    pub fn memset_async(&mut self, value: i32, stream: cudaStream_t) -> Result<(), String> {
        if !self.ptr.is_null() && self.size_bytes > 0 {
            let status = unsafe { cudaMemsetAsync(self.ptr, value, self.size_bytes, stream) };
            if status != CUDA_SUCCESS {
                return Err(format!("cudaMemsetAsync failed: {status}"));
            }
        }
        Ok(())
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
