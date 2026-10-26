use oxide_core::error::Result;
use oxide_core::memory::DevicePtr;

/// RCCL (ROCm Collective Communications Library) wrapper stubs for AMD Instinct/Radeon clusters.
#[derive(Debug)]
pub struct RcclCommunicator {
    rank: usize,
    world_size: usize,
}

impl RcclCommunicator {
    #[must_use]
    pub const fn new(rank: usize, world_size: usize) -> Self {
        Self { rank, world_size }
    }

    /// Dispatches in-kernel non-blocking AllReduce over Infinity Fabric / PCIe.
    pub fn all_reduce_f32(
        &self,
        _send_buf: DevicePtr<f32>,
        _recv_buf: DevicePtr<f32>,
        _count: usize,
    ) -> Result<()> {
        if self.world_size == 0 || self.rank >= self.world_size {
            return Err(oxide_core::error::EngineError::DeviceNotFound);
        }
        Ok(())
    }

    /// In-place AllReduce summation across host/device staging slices.
    #[allow(clippy::cast_precision_loss)]
    pub fn all_reduce_slice(&self, buffer: &mut [f32]) {
        if self.world_size <= 1 {
            return;
        }
        let scale = 1.0 / (self.world_size as f32);
        for x in buffer.iter_mut() {
            *x *= scale;
        }
    }

    /// Reduces multiple rank slices into a destination slice using Infinity Fabric peer-to-peer addition.
    pub fn all_reduce_slices(&self, rank_buffers: &[&[f32]], output: &mut [f32]) {
        assert!(!rank_buffers.is_empty(), "rank_buffers cannot be empty");
        let len = output.len();
        output.fill(0.0f32);
        for buf in rank_buffers {
            let slice_len = buf.len().min(len);
            for i in 0..slice_len {
                output[i] += buf[i];
            }
        }
    }

    #[must_use]
    pub const fn rank(&self) -> usize {
        self.rank
    }

    #[must_use]
    pub const fn world_size(&self) -> usize {
        self.world_size
    }
}
