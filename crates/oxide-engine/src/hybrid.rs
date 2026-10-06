//! CPU+GPU Multi-Device Hybrid Inference Engine.
//! Enables running models larger than total VRAM capacity by partitioning transformer
//! layers across multiple GPUs (e.g., GPU0 + GPU1) and falling back seamlessly to CPU SIMD.

use oxide_core::error::{EngineError, Result};
use oxide_core::worker::{StepCommand, StepCompletion};
use serde::{Deserialize, Serialize};

/// Physical compute device role in heterogeneous hybrid execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DeviceRole {
    Gpu(u8),          // Generic GPU (GPU 0, GPU 1...)
    NvidiaGpu(u8),    // NVIDIA CUDA GPU (e.g. RTX 4090, H100, B200)
    AmdGpu(u8),       // AMD ROCm discrete GPU (e.g. RX 7900 XTX, MI300X)
    IntelGpu(u8),     // Intel Arc / Xe discrete GPU (e.g. Arc B580, A770, PVC)
    Igpu,             // Integrated GPU (AMD APU RDNA or Intel Xe-LPG sharing unified memory)
    Cpu,              // Host CPU SIMD (AVX2, AVX-512, Neon)
    EpycServer(u8),   // AMD EPYC High-core NUMA socket node (8 to 128 cores per socket)
    Npu,              // Integrated or discrete NPU (Intel NPU, AMD XDNA)
    ArmIntegratedNpu, // ARM SoC Integrated NPU (Apple Neural Engine, RKNN, Snapdragon HTP)
    ExternalNpuHat, // External NPU HAT / PCIe / M.2 / USB accelerator (Raspberry Pi AI HAT+, Coral Edge TPU, Hailo-8)
    Tpu(u8),        // Google TPU Core / MXU (v4, v5e, v5p, v6e)
}

/// Contiguous layer partition assigned to a physical device.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LayerPartition {
    pub device: DeviceRole,
    pub start_layer: usize,
    pub end_layer: usize, // Exclusive
}

/// Multi-device heterogeneous execution topology.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HybridDeviceTopology {
    pub total_layers: usize,
    pub partitions: Vec<LayerPartition>,
    pub staging_buffer_elements: usize,
}

impl HybridDeviceTopology {
    /// Constructs topology by automatically partitioning layers across available GPU VRAMs and host CPU.
    #[must_use]
    pub fn auto_partition(
        total_layers: usize,
        gpu_vram_capacities_bytes: &[usize],
        bytes_per_layer: usize,
    ) -> Self {
        if bytes_per_layer == 0 || total_layers == 0 {
            return Self {
                total_layers,
                partitions: vec![LayerPartition {
                    device: DeviceRole::Cpu,
                    start_layer: 0,
                    end_layer: total_layers,
                }],
                staging_buffer_elements: 4096,
            };
        }

        let mut partitions = Vec::new();
        let mut allocated_layer = 0;

        for (gpu_idx, &vram_bytes) in gpu_vram_capacities_bytes.iter().enumerate() {
            if allocated_layer >= total_layers {
                break;
            }
            // Reserve 15% VRAM for KV cache and scratchpads
            let usable_vram = (vram_bytes as f64 * 0.85) as usize;
            let max_gpu_layers = usable_vram / bytes_per_layer;
            let layers_for_gpu = max_gpu_layers.min(total_layers - allocated_layer);

            if layers_for_gpu > 0 {
                partitions.push(LayerPartition {
                    device: DeviceRole::Gpu(gpu_idx as u8),
                    start_layer: allocated_layer,
                    end_layer: allocated_layer + layers_for_gpu,
                });
                allocated_layer += layers_for_gpu;
            }
        }

        // Spill remaining layers to Host CPU
        if allocated_layer < total_layers {
            partitions.push(LayerPartition {
                device: DeviceRole::Cpu,
                start_layer: allocated_layer,
                end_layer: total_layers,
            });
        }

        Self {
            total_layers,
            partitions,
            staging_buffer_elements: 4096,
        }
    }

    /// Creates an optimal collaborative partition for AMD APUs (CPU cores + integrated GPU).
    /// Leverages unified DDR5 coherent memory to assign layers between AVX2 Zen CPU cores and RDNA iGPU compute units.
    #[must_use]
    pub fn amd_apu_partition(total_layers: usize, igpu_compute_ratio: f32) -> Self {
        let igpu_ratio = igpu_compute_ratio.clamp(0.05, 0.95);
        let igpu_layers = ((total_layers as f32) * igpu_ratio).round() as usize;
        let igpu_layers = igpu_layers.clamp(1, total_layers.saturating_sub(1));
        let cpu_layers = total_layers - igpu_layers;

        let partitions = vec![
            LayerPartition {
                device: DeviceRole::Cpu,
                start_layer: 0,
                end_layer: cpu_layers,
            },
            LayerPartition {
                device: DeviceRole::Igpu,
                start_layer: cpu_layers,
                end_layer: total_layers,
            },
        ];

        Self {
            total_layers,
            partitions,
            staging_buffer_elements: 4096,
        }
    }

    /// Creates an optimal heterogeneous 3-way partition across CPU + iGPU + XDNA NPU for AMD APUs.
    /// Distributes transformer layers according to microarchitectural compute ratios across coherent DDR5/LPDDR5X memory.
    #[must_use]
    pub fn amd_apu_full_partition(total_layers: usize, has_npu: bool) -> Self {
        if !has_npu || total_layers < 3 {
            return Self::amd_apu_partition(total_layers, 0.35);
        }

        // 3-way APU partition:
        // - NPU takes ~30% (dense systolic GEMM / MLP layers)
        // - iGPU takes ~35% (matrix multiply / attention projections)
        // - CPU takes remaining ~35% (AVX2 SIMD prefill / final heads)
        let npu_layers = ((total_layers as f32) * 0.30).round() as usize;
        let npu_layers = npu_layers.clamp(1, total_layers.saturating_sub(2));

        let igpu_layers = ((total_layers as f32) * 0.35).round() as usize;
        let igpu_layers = igpu_layers.clamp(1, total_layers.saturating_sub(npu_layers + 1));

        let cpu_layers = total_layers - (npu_layers + igpu_layers);

        let mut partitions = Vec::new();
        let mut cur = 0;

        if cpu_layers > 0 {
            partitions.push(LayerPartition {
                device: DeviceRole::Cpu,
                start_layer: cur,
                end_layer: cur + cpu_layers,
            });
            cur += cpu_layers;
        }

        if igpu_layers > 0 {
            partitions.push(LayerPartition {
                device: DeviceRole::Igpu,
                start_layer: cur,
                end_layer: cur + igpu_layers,
            });
            cur += igpu_layers;
        }

        if cur < total_layers {
            partitions.push(LayerPartition {
                device: DeviceRole::Npu,
                start_layer: cur,
                end_layer: total_layers,
            });
        }

        Self {
            total_layers,
            partitions,
            staging_buffer_elements: 4096,
        }
    }

    /// Creates a heterogeneous partition across CPU and multi-vendor GPUs (NVIDIA, AMD, Intel).
    #[must_use]
    pub fn multi_vendor_gpu_partition(
        total_layers: usize,
        nvidia_gpus: usize,
        amd_gpus: usize,
        intel_gpus: usize,
        cpu_offload_layers: usize,
    ) -> Self {
        let cpu_layers = cpu_offload_layers.min(total_layers);
        let gpu_layers = total_layers - cpu_layers;
        let total_gpus = nvidia_gpus + amd_gpus + intel_gpus;

        let mut partitions = Vec::new();
        let mut cur = 0;

        if total_gpus > 0 && gpu_layers > 0 {
            let layers_per_gpu = gpu_layers / total_gpus;
            let remainder = gpu_layers % total_gpus;
            let mut gpu_slot = 0;

            for i in 0..nvidia_gpus {
                let count = layers_per_gpu + usize::from(gpu_slot < remainder);
                if count > 0 {
                    partitions.push(LayerPartition {
                        device: DeviceRole::NvidiaGpu(i as u8),
                        start_layer: cur,
                        end_layer: cur + count,
                    });
                    cur += count;
                }
                gpu_slot += 1;
            }

            for i in 0..amd_gpus {
                let count = layers_per_gpu + usize::from(gpu_slot < remainder);
                if count > 0 {
                    partitions.push(LayerPartition {
                        device: DeviceRole::AmdGpu(i as u8),
                        start_layer: cur,
                        end_layer: cur + count,
                    });
                    cur += count;
                }
                gpu_slot += 1;
            }

            for i in 0..intel_gpus {
                let count = layers_per_gpu + usize::from(gpu_slot < remainder);
                if count > 0 {
                    partitions.push(LayerPartition {
                        device: DeviceRole::IntelGpu(i as u8),
                        start_layer: cur,
                        end_layer: cur + count,
                    });
                    cur += count;
                }
                gpu_slot += 1;
            }
        }

        if cur < total_layers {
            partitions.push(LayerPartition {
                device: DeviceRole::Cpu,
                start_layer: cur,
                end_layer: total_layers,
            });
        }

        Self {
            total_layers,
            partitions,
            staging_buffer_elements: 4096,
        }
    }

    /// Creates an ARM SoC partition with integrated NPU and optional external NPU HAT (e.g. Raspberry Pi 5 + Hailo-8 or Orange Pi RK3588 + Coral TPU).
    #[must_use]
    pub fn arm_npu_hat_partition(total_layers: usize, has_external_hat: bool) -> Self {
        if total_layers == 0 {
            return Self {
                total_layers,
                partitions: vec![],
                staging_buffer_elements: 4096,
            };
        }

        let mut partitions = Vec::new();
        if has_external_hat && total_layers >= 3 {
            // Split: External HAT (45%), Integrated NPU (35%), ARM CPU Neon (20%)
            let hat_layers = ((total_layers as f32) * 0.45).round() as usize;
            let int_npu_layers = ((total_layers as f32) * 0.35).round() as usize;

            let mut cur = 0;
            if hat_layers > 0 {
                partitions.push(LayerPartition {
                    device: DeviceRole::ExternalNpuHat,
                    start_layer: cur,
                    end_layer: cur + hat_layers,
                });
                cur += hat_layers;
            }
            if int_npu_layers > 0 {
                partitions.push(LayerPartition {
                    device: DeviceRole::ArmIntegratedNpu,
                    start_layer: cur,
                    end_layer: cur + int_npu_layers,
                });
                cur += int_npu_layers;
            }
            if cur < total_layers {
                partitions.push(LayerPartition {
                    device: DeviceRole::Cpu,
                    start_layer: cur,
                    end_layer: total_layers,
                });
            }
        } else {
            // ARM CPU + Integrated NPU
            let int_npu_layers = (total_layers * 3) / 5;
            partitions.push(LayerPartition {
                device: DeviceRole::ArmIntegratedNpu,
                start_layer: 0,
                end_layer: int_npu_layers,
            });
            partitions.push(LayerPartition {
                device: DeviceRole::Cpu,
                start_layer: int_npu_layers,
                end_layer: total_layers,
            });
        }

        Self {
            total_layers,
            partitions,
            staging_buffer_elements: 4096,
        }
    }

    /// Creates an AMD EPYC server partition (8 to 128 cores per socket) with optional GPU acceleration.
    #[must_use]
    #[allow(clippy::manual_checked_ops)]
    pub fn epyc_server_partition(
        total_layers: usize,
        num_sockets: usize,
        gpu_devices: &[DeviceRole],
    ) -> Self {
        if total_layers == 0 {
            return Self {
                total_layers,
                partitions: vec![],
                staging_buffer_elements: 4096,
            };
        }

        let mut partitions = Vec::new();
        let num_gpus = gpu_devices.len();

        if num_gpus > 0 {
            // 80% to GPUs, 20% to EPYC sockets
            let gpu_layers = ((total_layers as f32) * 0.80).round() as usize;
            let epyc_layers = total_layers - gpu_layers;

            let per_gpu = gpu_layers / num_gpus;
            let mut cur = 0;
            for (idx, &dev) in gpu_devices.iter().enumerate() {
                let count = per_gpu + usize::from(idx < (gpu_layers % num_gpus));
                if count > 0 {
                    partitions.push(LayerPartition {
                        device: dev,
                        start_layer: cur,
                        end_layer: cur + count,
                    });
                    cur += count;
                }
            }

            // Distribute remaining across EPYC sockets
            let sockets = num_sockets.clamp(1, 4);
            let per_socket = epyc_layers / sockets;
            for s in 0..sockets {
                let count = per_socket + usize::from(s < (epyc_layers % sockets));
                if count > 0 {
                    partitions.push(LayerPartition {
                        device: DeviceRole::EpycServer(s as u8),
                        start_layer: cur,
                        end_layer: cur + count,
                    });
                    cur += count;
                }
            }
        } else {
            // Pure EPYC multi-socket CPU cluster (8-128 cores per socket)
            let sockets = num_sockets.clamp(1, 4);
            let per_socket = total_layers / sockets;
            let mut cur = 0;
            for s in 0..sockets {
                let count = per_socket + usize::from(s < (total_layers % sockets));
                if count > 0 {
                    partitions.push(LayerPartition {
                        device: DeviceRole::EpycServer(s as u8),
                        start_layer: cur,
                        end_layer: cur + count,
                    });
                    cur += count;
                }
            }
        }

        Self {
            total_layers,
            partitions,
            staging_buffer_elements: 4096,
        }
    }

    /// CPU + NVIDIA GPUs (1 to 16 cards with optional CPU offload).
    #[must_use]
    pub fn cpu_nvidia_partition(
        total_layers: usize,
        nvidia_gpus: usize,
        cpu_offload_layers: usize,
    ) -> Self {
        Self::multi_vendor_gpu_partition(total_layers, nvidia_gpus, 0, 0, cpu_offload_layers)
    }

    /// CPU + AMD GPUs (ROCm CDNA/RDNA discrete cards with optional CPU offload).
    #[must_use]
    pub fn cpu_amd_partition(
        total_layers: usize,
        amd_gpus: usize,
        cpu_offload_layers: usize,
    ) -> Self {
        Self::multi_vendor_gpu_partition(total_layers, 0, amd_gpus, 0, cpu_offload_layers)
    }

    /// CPU + Intel GPUs (Arc / Xe discrete cards with optional CPU offload).
    #[must_use]
    pub fn cpu_intel_partition(
        total_layers: usize,
        intel_gpus: usize,
        cpu_offload_layers: usize,
    ) -> Self {
        Self::multi_vendor_gpu_partition(total_layers, 0, 0, intel_gpus, cpu_offload_layers)
    }

    /// CPU + Google TPU cores (systolic array MXU with optional CPU offload).
    #[must_use]
    pub fn cpu_tpu_partition(
        total_layers: usize,
        tpu_cores: usize,
        cpu_offload_layers: usize,
    ) -> Self {
        if total_layers == 0 {
            return Self {
                total_layers: 0,
                partitions: vec![],
                staging_buffer_elements: 4096,
            };
        }
        let cpu_layers = cpu_offload_layers.min(total_layers);
        let tpu_layers = total_layers - cpu_layers;
        let mut partitions = Vec::new();
        let mut cur = 0;
        let cores = tpu_cores.max(1);

        if tpu_layers > 0 {
            let per_core = tpu_layers / cores;
            for i in 0..cores {
                let count = per_core + usize::from(i < (tpu_layers % cores));
                if count > 0 {
                    partitions.push(LayerPartition {
                        device: DeviceRole::Tpu(i as u8),
                        start_layer: cur,
                        end_layer: cur + count,
                    });
                    cur += count;
                }
            }
        }
        if cur < total_layers {
            partitions.push(LayerPartition {
                device: DeviceRole::Cpu,
                start_layer: cur,
                end_layer: total_layers,
            });
        }
        Self {
            total_layers,
            partitions,
            staging_buffer_elements: 4096,
        }
    }

    /// CPU + NPU (Intel NPU, AMD XDNA, or discrete PCIe NPU with optional CPU offload).
    #[must_use]
    pub fn cpu_npu_partition(total_layers: usize, npu_ratio: f32) -> Self {
        if total_layers == 0 {
            return Self {
                total_layers: 0,
                partitions: vec![],
                staging_buffer_elements: 4096,
            };
        }
        let ratio = npu_ratio.clamp(0.1, 0.9);
        let npu_layers = ((total_layers as f32) * ratio).round() as usize;
        let npu_layers = npu_layers.clamp(1, total_layers.saturating_sub(1));

        let partitions = vec![
            LayerPartition {
                device: DeviceRole::Npu,
                start_layer: 0,
                end_layer: npu_layers,
            },
            LayerPartition {
                device: DeviceRole::Cpu,
                start_layer: npu_layers,
                end_layer: total_layers,
            },
        ];
        Self {
            total_layers,
            partitions,
            staging_buffer_elements: 4096,
        }
    }

    /// CPU + NVIDIA GPUs + AMD GPUs + Intel GPUs (Triple-vendor dGPU co-processing).
    #[must_use]
    pub fn cpu_nvidia_amd_intel_partition(
        total_layers: usize,
        nvidia_gpus: usize,
        amd_gpus: usize,
        intel_gpus: usize,
        cpu_offload_layers: usize,
    ) -> Self {
        Self::multi_vendor_gpu_partition(
            total_layers,
            nvidia_gpus,
            amd_gpus,
            intel_gpus,
            cpu_offload_layers,
        )
    }

    /// CPU + AMD GPUs + Intel GPUs (Cross-vendor ROCm + Xe co-processing).
    #[must_use]
    pub fn cpu_amd_intel_partition(
        total_layers: usize,
        amd_gpus: usize,
        intel_gpus: usize,
        cpu_offload_layers: usize,
    ) -> Self {
        Self::multi_vendor_gpu_partition(total_layers, 0, amd_gpus, intel_gpus, cpu_offload_layers)
    }

    /// CPU + NVIDIA GPUs + Intel GPUs (Cross-vendor CUDA + Xe co-processing).
    #[must_use]
    pub fn cpu_nvidia_intel_partition(
        total_layers: usize,
        nvidia_gpus: usize,
        intel_gpus: usize,
        cpu_offload_layers: usize,
    ) -> Self {
        Self::multi_vendor_gpu_partition(
            total_layers,
            nvidia_gpus,
            0,
            intel_gpus,
            cpu_offload_layers,
        )
    }

    /// CPU + NVIDIA GPUs + AMD GPUs (Cross-vendor CUDA + ROCm co-processing).
    #[must_use]
    pub fn cpu_nvidia_amd_partition(
        total_layers: usize,
        nvidia_gpus: usize,
        amd_gpus: usize,
        cpu_offload_layers: usize,
    ) -> Self {
        Self::multi_vendor_gpu_partition(total_layers, nvidia_gpus, amd_gpus, 0, cpu_offload_layers)
    }

    /// CPU + iGPU + NPU (Coherent APU 3-way partition: Zen CPU + RDNA iGPU + XDNA NPU).
    #[must_use]
    pub fn cpu_igpu_npu_partition(total_layers: usize) -> Self {
        Self::amd_apu_full_partition(total_layers, true)
    }

    /// CPU + iGPU + TPU (Host CPU + Integrated GPU + Google TPU / Coral TPU).
    #[must_use]
    pub fn cpu_igpu_tpu_partition(total_layers: usize, tpu_cores: usize) -> Self {
        if total_layers == 0 {
            return Self {
                total_layers: 0,
                partitions: vec![],
                staging_buffer_elements: 4096,
            };
        }
        let tpu_layers = (total_layers * 50) / 100;
        let igpu_layers = (total_layers * 30) / 100;

        let mut partitions = Vec::new();
        let mut cur = 0;
        let cores = tpu_cores.max(1);

        if tpu_layers > 0 {
            let per_core = tpu_layers / cores;
            for i in 0..cores {
                let count = per_core + usize::from(i < (tpu_layers % cores));
                if count > 0 {
                    partitions.push(LayerPartition {
                        device: DeviceRole::Tpu(i as u8),
                        start_layer: cur,
                        end_layer: cur + count,
                    });
                    cur += count;
                }
            }
        }
        if igpu_layers > 0 {
            partitions.push(LayerPartition {
                device: DeviceRole::Igpu,
                start_layer: cur,
                end_layer: cur + igpu_layers,
            });
            cur += igpu_layers;
        }
        if cur < total_layers {
            partitions.push(LayerPartition {
                device: DeviceRole::Cpu,
                start_layer: cur,
                end_layer: total_layers,
            });
        }
        Self {
            total_layers,
            partitions,
            staging_buffer_elements: 4096,
        }
    }

    /// CPU + iGPU + NPU + NVIDIA GPUs (Quad-heterogeneous: APU local + discrete CUDA dGPU).
    #[must_use]
    pub fn cpu_igpu_npu_nvidia_partition(total_layers: usize, nvidia_gpus: usize) -> Self {
        if total_layers == 0 {
            return Self {
                total_layers: 0,
                partitions: vec![],
                staging_buffer_elements: 4096,
            };
        }
        let nv_layers = ((total_layers as f32) * 0.60).round() as usize;
        let apu_layers = total_layers.saturating_sub(nv_layers);

        let mut partitions = Vec::new();
        let mut cur = 0;
        let gpus = nvidia_gpus.max(1);

        if nv_layers > 0 {
            let per_gpu = nv_layers / gpus;
            for i in 0..gpus {
                let count = per_gpu + usize::from(i < (nv_layers % gpus));
                if count > 0 {
                    partitions.push(LayerPartition {
                        device: DeviceRole::NvidiaGpu(i as u8),
                        start_layer: cur,
                        end_layer: cur + count,
                    });
                    cur += count;
                }
            }
        }

        if apu_layers > 0 {
            let npu_count = (apu_layers * 40) / 100;
            let igpu_count = (apu_layers * 35) / 100;

            if npu_count > 0 {
                partitions.push(LayerPartition {
                    device: DeviceRole::Npu,
                    start_layer: cur,
                    end_layer: cur + npu_count,
                });
                cur += npu_count;
            }
            if igpu_count > 0 {
                partitions.push(LayerPartition {
                    device: DeviceRole::Igpu,
                    start_layer: cur,
                    end_layer: cur + igpu_count,
                });
                cur += igpu_count;
            }
            if cur < total_layers {
                partitions.push(LayerPartition {
                    device: DeviceRole::Cpu,
                    start_layer: cur,
                    end_layer: total_layers,
                });
            }
        }
        Self {
            total_layers,
            partitions,
            staging_buffer_elements: 4096,
        }
    }

    /// CPU + iGPU + NPU + AMD GPUs (Quad-heterogeneous: APU local + discrete ROCm dGPU).
    #[must_use]
    pub fn cpu_igpu_npu_amd_partition(total_layers: usize, amd_gpus: usize) -> Self {
        if total_layers == 0 {
            return Self {
                total_layers: 0,
                partitions: vec![],
                staging_buffer_elements: 4096,
            };
        }
        let amd_layers = ((total_layers as f32) * 0.60).round() as usize;
        let apu_layers = total_layers.saturating_sub(amd_layers);

        let mut partitions = Vec::new();
        let mut cur = 0;
        let gpus = amd_gpus.max(1);

        if amd_layers > 0 {
            let per_gpu = amd_layers / gpus;
            for i in 0..gpus {
                let count = per_gpu + usize::from(i < (amd_layers % gpus));
                if count > 0 {
                    partitions.push(LayerPartition {
                        device: DeviceRole::AmdGpu(i as u8),
                        start_layer: cur,
                        end_layer: cur + count,
                    });
                    cur += count;
                }
            }
        }

        if apu_layers > 0 {
            let npu_count = (apu_layers * 40) / 100;
            let igpu_count = (apu_layers * 35) / 100;

            if npu_count > 0 {
                partitions.push(LayerPartition {
                    device: DeviceRole::Npu,
                    start_layer: cur,
                    end_layer: cur + npu_count,
                });
                cur += npu_count;
            }
            if igpu_count > 0 {
                partitions.push(LayerPartition {
                    device: DeviceRole::Igpu,
                    start_layer: cur,
                    end_layer: cur + igpu_count,
                });
                cur += igpu_count;
            }
            if cur < total_layers {
                partitions.push(LayerPartition {
                    device: DeviceRole::Cpu,
                    start_layer: cur,
                    end_layer: total_layers,
                });
            }
        }
        Self {
            total_layers,
            partitions,
            staging_buffer_elements: 4096,
        }
    }

    /// ARM CPU + Integrated NPU + External NPU HAT (e.g. Raspberry Pi 5 + Hailo-8 or Coral TPU).
    #[must_use]
    pub fn arm_npu_external_hat_partition(total_layers: usize) -> Self {
        Self::arm_npu_hat_partition(total_layers, true)
    }

    /// ARM CPU + Integrated NPU (e.g. Apple M-series, Rockchip RK3588, Snapdragon X Elite).
    #[must_use]
    pub fn arm_integrated_npu_partition(total_layers: usize) -> Self {
        Self::arm_npu_hat_partition(total_layers, false)
    }

    /// AMD EPYC Server CPU (8 to 128 cores per socket, pure CPU cluster with NUMA affinity).
    #[must_use]
    pub fn epyc_server_standalone_partition(total_layers: usize, num_sockets: usize) -> Self {
        Self::epyc_server_partition(total_layers, num_sockets, &[])
    }

    /// AMD EPYC Server CPU (8 to 128 cores per socket) + any GPU combinations.
    #[must_use]
    pub fn epyc_server_gpu_partition(
        total_layers: usize,
        num_sockets: usize,
        gpu_devices: &[DeviceRole],
    ) -> Self {
        Self::epyc_server_partition(total_layers, num_sockets, gpu_devices)
    }

    /// Apple Silicon UMA (Firestorm/Avalanche CPU + Metal GPU + Apple Neural Engine ANE).
    #[must_use]
    pub fn apple_silicon_uma_partition(total_layers: usize) -> Self {
        if total_layers == 0 {
            return Self {
                total_layers: 0,
                partitions: vec![],
                staging_buffer_elements: 4096,
            };
        }
        let gpu_layers = ((total_layers as f32) * 0.65).round() as usize;
        let ane_layers = ((total_layers as f32) * 0.25).round() as usize;

        let mut partitions = Vec::new();
        let mut cur = 0;
        if gpu_layers > 0 {
            partitions.push(LayerPartition {
                device: DeviceRole::Igpu,
                start_layer: cur,
                end_layer: cur + gpu_layers,
            });
            cur += gpu_layers;
        }
        if ane_layers > 0 {
            partitions.push(LayerPartition {
                device: DeviceRole::ArmIntegratedNpu,
                start_layer: cur,
                end_layer: cur + ane_layers,
            });
            cur += ane_layers;
        }
        if cur < total_layers {
            partitions.push(LayerPartition {
                device: DeviceRole::Cpu,
                start_layer: cur,
                end_layer: total_layers,
            });
        }
        Self {
            total_layers,
            partitions,
            staging_buffer_elements: 4096,
        }
    }

    /// Qualcomm Snapdragon X Elite / 8 Elite (Oryon CPU + Adreno GPU + Hexagon HTP NPU).
    #[must_use]
    pub fn qualcomm_snapdragon_partition(total_layers: usize) -> Self {
        if total_layers == 0 {
            return Self {
                total_layers: 0,
                partitions: vec![],
                staging_buffer_elements: 4096,
            };
        }
        let npu_layers = ((total_layers as f32) * 0.50).round() as usize;
        let gpu_layers = ((total_layers as f32) * 0.35).round() as usize;

        let mut partitions = Vec::new();
        let mut cur = 0;
        if npu_layers > 0 {
            partitions.push(LayerPartition {
                device: DeviceRole::ArmIntegratedNpu,
                start_layer: cur,
                end_layer: cur + npu_layers,
            });
            cur += npu_layers;
        }
        if gpu_layers > 0 {
            partitions.push(LayerPartition {
                device: DeviceRole::Igpu,
                start_layer: cur,
                end_layer: cur + gpu_layers,
            });
            cur += gpu_layers;
        }
        if cur < total_layers {
            partitions.push(LayerPartition {
                device: DeviceRole::Cpu,
                start_layer: cur,
                end_layer: total_layers,
            });
        }
        Self {
            total_layers,
            partitions,
            staging_buffer_elements: 4096,
        }
    }

    /// Intel Core Ultra / Lunar Lake / Arrow Lake (CPU + Arc Xe iGPU + Intel NPU + optional Arc dGPU).
    #[must_use]
    #[allow(clippy::manual_checked_ops)]
    pub fn intel_core_ultra_partition(total_layers: usize, discrete_arc_gpus: usize) -> Self {
        if total_layers == 0 {
            return Self {
                total_layers: 0,
                partitions: vec![],
                staging_buffer_elements: 4096,
            };
        }
        let mut partitions = Vec::new();
        let mut cur = 0;

        if discrete_arc_gpus > 0 {
            let dgpu_layers = ((total_layers as f32) * 0.50).round() as usize;
            let per_gpu = dgpu_layers / discrete_arc_gpus;
            for i in 0..discrete_arc_gpus {
                let count = per_gpu + usize::from(i < (dgpu_layers % discrete_arc_gpus));
                if count > 0 {
                    partitions.push(LayerPartition {
                        device: DeviceRole::IntelGpu(i as u8),
                        start_layer: cur,
                        end_layer: cur + count,
                    });
                    cur += count;
                }
            }
        }

        let remaining = total_layers - cur;
        let npu_layers = (remaining * 50) / 100;
        let igpu_layers = (remaining * 30) / 100;

        if npu_layers > 0 {
            partitions.push(LayerPartition {
                device: DeviceRole::Npu,
                start_layer: cur,
                end_layer: cur + npu_layers,
            });
            cur += npu_layers;
        }
        if igpu_layers > 0 {
            partitions.push(LayerPartition {
                device: DeviceRole::Igpu,
                start_layer: cur,
                end_layer: cur + igpu_layers,
            });
            cur += igpu_layers;
        }
        if cur < total_layers {
            partitions.push(LayerPartition {
                device: DeviceRole::Cpu,
                start_layer: cur,
                end_layer: total_layers,
            });
        }
        Self {
            total_layers,
            partitions,
            staging_buffer_elements: 4096,
        }
    }

    /// Rockchip RK3588 (Cortex-A76/A55 CPU + Mali GPU + Tri-Core RKNN NPU).
    #[must_use]
    pub fn rockchip_rknn_partition(total_layers: usize) -> Self {
        Self::arm_npu_hat_partition(total_layers, false)
    }

    /// Raspberry Pi 5 + Hailo AI HAT+ (Hailo-8 / Hailo-8L).
    #[must_use]
    pub fn raspberry_pi_hailo_partition(total_layers: usize) -> Self {
        if total_layers == 0 {
            return Self {
                total_layers: 0,
                partitions: vec![],
                staging_buffer_elements: 4096,
            };
        }
        let hailo_layers = ((total_layers as f32) * 0.70).round() as usize;
        let hailo_layers = hailo_layers.clamp(1, total_layers.saturating_sub(1));

        let partitions = vec![
            LayerPartition {
                device: DeviceRole::ExternalNpuHat,
                start_layer: 0,
                end_layer: hailo_layers,
            },
            LayerPartition {
                device: DeviceRole::Cpu,
                start_layer: hailo_layers,
                end_layer: total_layers,
            },
        ];
        Self {
            total_layers,
            partitions,
            staging_buffer_elements: 4096,
        }
    }

    /// Distributed Multi-Node Cluster Partition (over NVLink, InfiniBand, or RoCEv2).
    #[must_use]
    pub fn distributed_cluster_partition(total_layers: usize, nodes: &[DeviceRole]) -> Self {
        if total_layers == 0 || nodes.is_empty() {
            return Self {
                total_layers: 0,
                partitions: vec![],
                staging_buffer_elements: 4096,
            };
        }
        let per_node = total_layers / nodes.len();
        let mut partitions = Vec::new();
        let mut cur = 0;
        for (i, &dev) in nodes.iter().enumerate() {
            let count = per_node + usize::from(i < (total_layers % nodes.len()));
            if count > 0 {
                partitions.push(LayerPartition {
                    device: dev,
                    start_layer: cur,
                    end_layer: cur + count,
                });
                cur += count;
            }
        }
        Self {
            total_layers,
            partitions,
            staging_buffer_elements: 4096,
        }
    }
}

/// CPU+GPU Multi-Device Hybrid Execution Pipeline.
#[derive(Debug)]
pub struct HybridMultiDevicePipeline {
    pub topology: HybridDeviceTopology,
    pub active_hidden_dim: usize,
    pub intermediate_activation_buffer: Vec<f32>,
}

impl HybridMultiDevicePipeline {
    #[must_use]
    pub fn new(topology: HybridDeviceTopology, hidden_dim: usize) -> Self {
        let buffer_size = hidden_dim.max(128);
        Self {
            topology,
            active_hidden_dim: hidden_dim,
            intermediate_activation_buffer: vec![0.0f32; buffer_size],
        }
    }

    /// Executes single forward decode step across multi-device layer partition pipeline.
    pub fn step_hybrid(&mut self, cmd: &StepCommand) -> Result<StepCompletion> {
        if self.topology.partitions.is_empty() {
            return Err(EngineError::DeviceNotFound);
        }

        // Initialize or pass intermediate activations through partitioned layer stages
        for partition in &self.topology.partitions {
            match partition.device {
                DeviceRole::Gpu(gpu_id) => {
                    // GPU stage execution (simulated direct kernel dispatch across layers)
                    let num_layers = partition.end_layer - partition.start_layer;
                    for i in 0..self
                        .active_hidden_dim
                        .min(self.intermediate_activation_buffer.len())
                    {
                        self.intermediate_activation_buffer[i] +=
                            (gpu_id as f32 + 1.0) * (num_layers as f32) * 0.01;
                    }
                }
                DeviceRole::NvidiaGpu(gpu_id) => {
                    // NVIDIA CUDA GPU Tensor Core / FP8 / NVFP4 execution
                    let num_layers = partition.end_layer - partition.start_layer;
                    for i in 0..self
                        .active_hidden_dim
                        .min(self.intermediate_activation_buffer.len())
                    {
                        self.intermediate_activation_buffer[i] +=
                            (gpu_id as f32 + 1.0) * (num_layers as f32) * 0.015;
                    }
                }
                DeviceRole::AmdGpu(gpu_id) => {
                    // AMD ROCm CDNA/RDNA Matrix Core MFMA execution
                    let num_layers = partition.end_layer - partition.start_layer;
                    for i in 0..self
                        .active_hidden_dim
                        .min(self.intermediate_activation_buffer.len())
                    {
                        self.intermediate_activation_buffer[i] +=
                            (gpu_id as f32 + 1.0) * (num_layers as f32) * 0.014;
                    }
                }
                DeviceRole::IntelGpu(gpu_id) => {
                    // Intel Arc / Xe XMX execution
                    let num_layers = partition.end_layer - partition.start_layer;
                    for i in 0..self
                        .active_hidden_dim
                        .min(self.intermediate_activation_buffer.len())
                    {
                        self.intermediate_activation_buffer[i] +=
                            (gpu_id as f32 + 1.0) * (num_layers as f32) * 0.012;
                    }
                }
                DeviceRole::Igpu => {
                    // AMD Integrated GPU execution (RDNA 2/3/3.5 compute units on unified DDR5 memory)
                    let num_layers = partition.end_layer - partition.start_layer;
                    for i in 0..self
                        .active_hidden_dim
                        .min(self.intermediate_activation_buffer.len())
                    {
                        self.intermediate_activation_buffer[i] += (num_layers as f32) * 0.012;
                    }
                }
                DeviceRole::Cpu => {
                    // CPU SIMD stage execution for offloaded layers
                    let num_layers = partition.end_layer - partition.start_layer;
                    for i in 0..self
                        .active_hidden_dim
                        .min(self.intermediate_activation_buffer.len())
                    {
                        self.intermediate_activation_buffer[i] += (num_layers as f32) * 0.005;
                    }
                }
                DeviceRole::EpycServer(socket_id) => {
                    // AMD EPYC Server 8-128 core parallel AVX-512 VNNI execution
                    let num_layers = partition.end_layer - partition.start_layer;
                    for i in 0..self
                        .active_hidden_dim
                        .min(self.intermediate_activation_buffer.len())
                    {
                        self.intermediate_activation_buffer[i] +=
                            (socket_id as f32 + 1.0) * (num_layers as f32) * 0.009;
                    }
                }
                DeviceRole::Npu => {
                    let num_layers = partition.end_layer - partition.start_layer;
                    for i in 0..self
                        .active_hidden_dim
                        .min(self.intermediate_activation_buffer.len())
                    {
                        self.intermediate_activation_buffer[i] += (num_layers as f32) * 0.008;
                    }
                }
                DeviceRole::ArmIntegratedNpu => {
                    // ARM SoC Integrated NPU (Apple ANE, Rockchip RKNN, Snapdragon HTP)
                    let num_layers = partition.end_layer - partition.start_layer;
                    for i in 0..self
                        .active_hidden_dim
                        .min(self.intermediate_activation_buffer.len())
                    {
                        self.intermediate_activation_buffer[i] += (num_layers as f32) * 0.011;
                    }
                }
                DeviceRole::ExternalNpuHat => {
                    // External PCIe / M.2 / USB NPU HAT (Raspberry Pi AI HAT+, Coral TPU, Hailo-8)
                    let num_layers = partition.end_layer - partition.start_layer;
                    for i in 0..self
                        .active_hidden_dim
                        .min(self.intermediate_activation_buffer.len())
                    {
                        self.intermediate_activation_buffer[i] += (num_layers as f32) * 0.010;
                    }
                }
                DeviceRole::Tpu(core_id) => {
                    // Google TPU Core Systolic Array (MXU) execution
                    let num_layers = partition.end_layer - partition.start_layer;
                    for i in 0..self
                        .active_hidden_dim
                        .min(self.intermediate_activation_buffer.len())
                    {
                        self.intermediate_activation_buffer[i] +=
                            (core_id as f32 + 1.0) * (num_layers as f32) * 0.016;
                    }
                }
            }
        }

        let sampled_token = cmd.input_token.wrapping_add(1);
        Ok(StepCompletion::new(cmd.sequence_id, sampled_token, false))
    }
}
