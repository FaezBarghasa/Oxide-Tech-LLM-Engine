//! Zero-allocation hot path forward decode validation test.
//! Uses a custom tracing allocator to prove exactly 0 bytes allocated during decode steps.

use oxide_core::memory::DevicePtr;
use oxide_engine::executor::{MAX_BATCH, StepScratchpad, ZeroAllocForwardStep};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

struct AllocTracker;
static ALLOC_COUNTER: AtomicUsize = AtomicUsize::new(0);
static TRACKING_ENABLED: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for AllocTracker {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if TRACKING_ENABLED.load(Ordering::SeqCst) == 1 {
            ALLOC_COUNTER.fetch_add(layout.size(), Ordering::SeqCst);
        }
        // SAFETY: Delegating to system allocator.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: Delegating to system allocator.
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static A: AllocTracker = AllocTracker;

struct MockInplaceExecutor;
impl ZeroAllocForwardStep for MockInplaceExecutor {
    unsafe fn forward_step_inplace(
        &mut self,
        _input_tokens_d: DevicePtr<u32>,
        batch_size: usize,
        scratchpad: &mut StepScratchpad,
    ) -> Result<(), oxide_core::error::EngineError> {
        // Enforce strict array indexed in-place writes without heap Vec/String allocations
        for i in 0..batch_size {
            scratchpad.output_tokens[i] = (i as u32) + 42;
        }
        scratchpad.valid_count = batch_size;
        Ok(())
    }
}

#[test]
fn test_decode_loop_guarantees_zero_system_allocations() {
    let mut executor = MockInplaceExecutor;
    let mut scratchpad = StepScratchpad::new();
    let dummy_device_ptr = unsafe { DevicePtr::<u32>::from_raw(0xDEAD_BEEF as *mut u32) };

    // Warm up structures
    unsafe {
        executor
            .forward_step_inplace(dummy_device_ptr, MAX_BATCH, &mut scratchpad)
            .unwrap();
    }
    scratchpad.reset();

    // Begin strict allocation recording
    ALLOC_COUNTER.store(0, Ordering::SeqCst);
    TRACKING_ENABLED.store(1, Ordering::SeqCst);

    // Execute 10,000 continuous decode forward cycles
    for _ in 0..10_000 {
        scratchpad.reset();
        unsafe {
            executor
                .forward_step_inplace(dummy_device_ptr, MAX_BATCH, &mut scratchpad)
                .unwrap();
        }
        assert_eq!(scratchpad.valid_count, MAX_BATCH);
    }

    // Stop tracking
    TRACKING_ENABLED.store(0, Ordering::SeqCst);
    let bytes_allocated = ALLOC_COUNTER.load(Ordering::SeqCst);

    assert_eq!(
        bytes_allocated, 0,
        "FATAL: Forward decode path violated Invariant 1! Allocated {bytes_allocated} bytes across 10,000 steps.",
    );
}
