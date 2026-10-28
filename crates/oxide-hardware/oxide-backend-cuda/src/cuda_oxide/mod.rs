//! `cuda-oxide`: Safe, zero-cost Rust abstractions for CUDA device execution
//! modeled after `NVlabs/cuda-rust` and `cuda-oxide`.
//!
//! Provides compile-time borrow checking across device partitions via [`DisjointSlice`],
//! unified kernel launching, fused activations (SwiGLU, GeGLU), RMSNorm, and on-device
//! argmax reduction directly on NVIDIA dGPU streams.

#![allow(clippy::not_unsafe_ptr_arg_deref)]

use crate::driver::{self, cudaStream_t};
use oxide_core::error::{EngineError, Result};
use std::marker::PhantomData;

/// Non-overlapping, borrow-checked slice of GPU device memory.
/// Guarantees at compile time that concurrent kernel partitions cannot alias.
#[derive(Debug)]
pub struct DisjointSlice<'a, T> {
    ptr: *mut T,
    len: usize,
    _marker: PhantomData<&'a mut [T]>,
}

unsafe impl<T: Send> Send for DisjointSlice<'_, T> {}
unsafe impl<T: Sync> Sync for DisjointSlice<'_, T> {}

impl<'a, T> DisjointSlice<'a, T> {
    /// Creates a disjoint slice from a raw device pointer and element count.
    ///
    /// # Safety
    /// The caller must ensure `ptr` points to valid GPU device memory of at least `len` elements,
    /// and that no other active `DisjointSlice` or mutable reference aliases this memory range.
    #[must_use]
    pub const unsafe fn new(ptr: *mut T, len: usize) -> Self {
        Self {
            ptr,
            len,
            _marker: PhantomData,
        }
    }

    /// Number of elements in the disjoint device slice.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Whether the device slice is empty.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the raw device pointer for kernel launch.
    #[must_use]
    pub const fn as_ptr(&self) -> *const T {
        self.ptr
    }

    /// Returns the mutable raw device pointer for kernel launch.
    #[must_use]
    pub const fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }

    /// Splits the disjoint slice into two non-overlapping halves at `mid`.
    /// Panics if `mid > len`.
    #[must_use]
    pub fn split_at_mut(self, mid: usize) -> (DisjointSlice<'a, T>, DisjointSlice<'a, T>) {
        assert!(
            mid <= self.len,
            "split index {mid} out of bounds for slice of len {}",
            self.len
        );
        let right_ptr = unsafe { self.ptr.add(mid) };
        let right_len = self.len - mid;
        (unsafe { DisjointSlice::new(self.ptr, mid) }, unsafe {
            DisjointSlice::new(right_ptr, right_len)
        })
    }
}

/// Dispatches fused SwiGLU activation: `out[i] = (gate[i] / (1 + exp(-gate[i]))) * up[i]`.
pub fn launch_fused_swiglu(
    mut out: DisjointSlice<'_, f32>,
    gate: &DisjointSlice<'_, f32>,
    up: &DisjointSlice<'_, f32>,
    stream: cudaStream_t,
) -> Result<()> {
    let n = out.len().min(gate.len()).min(up.len());
    if n == 0 {
        return Ok(());
    }
    let ret = unsafe {
        driver::launch_cuda_swiglu(
            out.as_mut_ptr(),
            gate.as_ptr(),
            up.as_ptr(),
            n as i32,
            stream,
        )
    };
    if ret != 0 {
        Err(EngineError::BackendError(format!(
            "CUDA fused SwiGLU kernel failed with code {ret}"
        )))
    } else {
        Ok(())
    }
}

/// Dispatches fused GeGLU activation (Gemma): `out[i] = GELU(gate[i]) * up[i]`.
pub fn launch_fused_geglu(
    mut out: DisjointSlice<'_, f32>,
    gate: &DisjointSlice<'_, f32>,
    up: &DisjointSlice<'_, f32>,
    stream: cudaStream_t,
) -> Result<()> {
    let n = out.len().min(gate.len()).min(up.len());
    if n == 0 {
        return Ok(());
    }
    let ret = unsafe {
        driver::launch_cuda_gelu(
            out.as_mut_ptr(),
            gate.as_ptr(),
            up.as_ptr(),
            n as i32,
            stream,
        )
    };
    if ret != 0 {
        Err(EngineError::BackendError(format!(
            "CUDA fused GeGLU kernel failed with code {ret}"
        )))
    } else {
        Ok(())
    }
}

/// Dispatches fused RMSNorm kernel across `hidden_dim` with optional Gemma `1.0 + w` weight offset.
pub fn launch_fused_rmsnorm(
    mut out: DisjointSlice<'_, f32>,
    input: &DisjointSlice<'_, f32>,
    weight: Option<&DisjointSlice<'_, f32>>,
    hidden_dim: usize,
    eps: f32,
    weight_offset: f32,
    stream: cudaStream_t,
) -> Result<()> {
    let num_tokens = (input.len() / hidden_dim.max(1)) as i32;
    if num_tokens <= 0 {
        return Ok(());
    }
    let weight_ptr = weight.map_or(std::ptr::null(), DisjointSlice::as_ptr);
    let ret = unsafe {
        driver::launch_cuda_rmsnorm(
            out.as_mut_ptr(),
            input.as_ptr(),
            weight_ptr,
            num_tokens,
            hidden_dim as i32,
            eps,
            weight_offset,
            stream,
        )
    };
    if ret != 0 {
        Err(EngineError::BackendError(format!(
            "CUDA fused RMSNorm kernel failed with code {ret}"
        )))
    } else {
        Ok(())
    }
}

/// Dispatches on-device vocabulary argmax reduction kernel.
/// Computes greedy token selection directly in GPU registers, bypassing 1MB PCIe transfers.
pub fn launch_on_device_argmax(
    logits: &DisjointSlice<'_, f32>,
    mut out_token_val: DisjointSlice<'_, u32>,
    stream: cudaStream_t,
) -> Result<()> {
    if logits.is_empty() || out_token_val.len() < 2 {
        return Err(EngineError::BackendError(
            "Invalid buffer dimensions for on-device argmax".to_string(),
        ));
    }
    let d_tok_ptr = out_token_val.as_mut_ptr();
    let d_val_ptr = unsafe { d_tok_ptr.add(1) as *mut f32 };
    let ret = unsafe {
        driver::launch_cuda_argmax(
            logits.as_ptr(),
            logits.len() as i32,
            d_tok_ptr,
            d_val_ptr,
            1,
            stream,
        )
    };
    if ret != 0 {
        Err(EngineError::BackendError(format!(
            "CUDA on-device argmax failed with code {ret}"
        )))
    } else {
        Ok(())
    }
}
